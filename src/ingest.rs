use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use tokio_postgres::Client;
use walkdir::WalkDir;

use crate::chunk::{chunk_markdown, sha256_hex, title_of};
use crate::embed::{Embedder, vector_literal};
use crate::token::term_freqs;

#[derive(Debug, Default)]
pub struct IngestStats {
    pub files_seen: usize,
    pub notes_written: usize,
    pub notes_skipped: usize,
    pub chunks_written: usize,
    pub warnings: Vec<String>,
}

pub async fn ingest_dir(
    client: &mut Client,
    embedder: &dyn Embedder,
    root: &Path,
    repo: Option<&Path>,
    max_chars: usize,
) -> Result<IngestStats> {
    let commit = repo.map(git_head).unwrap_or(None);

    let model_id = embedder.model_id();
    let mut stats = IngestStats::default();
    let files = collect_markdown(root);
    stats.files_seen = files.len();

    for file in files {
        let rel = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .to_string_lossy()
            .to_string();

        let src = match std::fs::read_to_string(&file) {
            Ok(s) => s,
            Err(e) => {
                stats.warnings.push(format!("跳过 {rel}: 读取失败 ({e})"));
                continue;
            }
        };
        let content_hash = sha256_hex(&src);

        let existing = client
            .query_opt(
                "select id, content_hash from notes where path = $1",
                &[&rel],
            )
            .await
            .context("查询已有笔记失败")?;

        let mut backfill_note: Option<i64> = None;
        if let Some(row) = &existing {
            let note_id: i64 = row.get(0);
            let old: String = row.get(1);
            if old == content_hash {
                // 内容没变，但换了嵌入模型时仍要补该模型的向量。
                let has_model = client
                    .query_opt(
                        "select 1
                         from embeddings e
                         join chunks c on c.id = e.chunk_id
                         where c.note_id = $1 and e.model = $2
                         limit 1",
                        &[&note_id, &model_id],
                    )
                    .await
                    .context("检查模型向量失败")?;
                if has_model.is_some() {
                    stats.notes_skipped += 1;
                    continue;
                }
                backfill_note = Some(note_id);
            }
        }

        // 只补新模型向量，保持 chunk id 不变（避免误伤记忆 provenance）。
        if let Some(note_id) = backfill_note {
            let rows = client
                .query(
                    "select id, heading, text from chunks where note_id = $1 order by ord",
                    &[&note_id],
                )
                .await
                .context("读取旧 chunk 失败")?;
            if !rows.is_empty() {
                let texts: Vec<String> = rows
                    .iter()
                    .map(|r| {
                        let h: String = r.get(1);
                        let t: String = r.get(2);
                        if h.is_empty() { t } else { format!("{h}\n{t}") }
                    })
                    .collect();
                let vectors = embedder
                    .embed(&texts)
                    .await
                    .with_context(|| format!("补模型向量失败: {rel}"))?;
                let tx = client.transaction().await.context("开启事务失败")?;
                for (r, vec) in rows.iter().zip(vectors.iter()) {
                    let chunk_id: i64 = r.get(0);
                    let lit = vector_literal(vec);
                    tx.execute(
                        "insert into embeddings (chunk_id, model, dim, vec)
                         values ($1, $2, $3, ($4::text)::vector)
                         on conflict (chunk_id, model) do update
                           set dim = excluded.dim, vec = excluded.vec",
                        &[&chunk_id, &model_id, &(vec.len() as i32), &lit],
                    )
                    .await
                    .context("补写 embedding 失败")?;
                }
                tx.commit().await.context("提交事务失败")?;
                stats.chunks_written += rows.len();
            }
            stats.notes_written += 1;
            continue;
        }

        let tx = client.transaction().await.context("开启事务失败")?;

        let fallback = file
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| rel.clone());
        let title = title_of(&src, &fallback);

        let row = tx
            .query_one(
                "insert into notes (path, title, content_hash, commit_hash)
                 values ($1, $2, $3, $4)
                 on conflict (path) do update
                   set title = excluded.title,
                       content_hash = excluded.content_hash,
                       commit_hash = excluded.commit_hash,
                       updated_at = now()
                 returning id",
                &[&rel, &title, &content_hash, &commit],
            )
            .await
            .context("写入 note 失败")?;
        let note_id: i64 = row.get(0);

        let chunks = chunk_markdown(&src, max_chars);
        if chunks.is_empty() {
            tx.execute("delete from chunks where note_id = $1", &[&note_id])
                .await
                .context("清理旧 chunk 失败")?;
            tx.commit().await.context("提交事务失败")?;
            stats.notes_written += 1;
            continue;
        }

        // 只复用“内容哈希相同且已有当前模型向量”的 chunk，其余重嵌。
        let reusable_rows = tx
            .query(
                "select c.id, c.chunk_hash
                 from chunks c
                 where c.note_id = $1
                   and exists (select 1 from embeddings e
                               where e.chunk_id = c.id and e.model = $2)",
                &[&note_id, &model_id],
            )
            .await
            .context("读取可复用 chunk 失败")?;
        let mut reusable: HashMap<String, Vec<i64>> = HashMap::new();
        for r in reusable_rows {
            reusable.entry(r.get(1)).or_default().push(r.get(0));
        }

        let mut keep: Vec<(usize, i64)> = Vec::new();
        let mut new_idx: Vec<usize> = Vec::new();
        for (i, c) in chunks.iter().enumerate() {
            if let Some(ids) = reusable.get_mut(&c.hash)
                && let Some(id) = ids.pop()
            {
                keep.push((i, id));
                continue;
            }
            new_idx.push(i);
        }

        let to_embed: Vec<String> = new_idx
            .iter()
            .map(|&i| {
                let c = &chunks[i];
                if c.heading.is_empty() {
                    c.text.clone()
                } else {
                    format!("{}\n{}", c.heading, c.text)
                }
            })
            .collect();
        let vectors = if to_embed.is_empty() {
            Vec::new()
        } else {
            embedder
                .embed(&to_embed)
                .await
                .with_context(|| format!("嵌入失败: {rel}"))?
        };
        if vectors.len() != new_idx.len() {
            anyhow::bail!("嵌入数量与新增切块数量不一致: {rel}");
        }

        // 先把旧 ord 挪开，避免唯一约束冲突，再摆回保留块。
        tx.execute(
            "update chunks set ord = ord + 1000000 where note_id = $1",
            &[&note_id],
        )
        .await
        .context("临时移动 chunk ord 失败")?;
        for (idx, id) in &keep {
            tx.execute(
                "update chunks set ord = $1 where id = $2",
                &[&chunks[*idx].ord, id],
            )
            .await
            .context("更新保留 chunk 顺序失败")?;
        }
        let keep_ids: Vec<i64> = keep.iter().map(|(_, id)| *id).collect();
        if keep_ids.is_empty() {
            tx.execute("delete from chunks where note_id = $1", &[&note_id])
                .await
                .context("清理旧 chunk 失败")?;
        } else {
            tx.execute(
                "delete from chunks where note_id = $1 and id <> all($2::bigint[])",
                &[&note_id, &keep_ids],
            )
            .await
            .context("删除被移除的 chunk 失败")?;
        }

        for (vec_pos, &i) in new_idx.iter().enumerate() {
            let chunk = &chunks[i];
            let vec = &vectors[vec_pos];
            let lex_src = if chunk.heading.is_empty() {
                chunk.text.clone()
            } else {
                format!("{}\n{}", chunk.heading, chunk.text)
            };
            let (terms, token_count) = term_freqs(&lex_src);

            let row = tx
                .query_one(
                    "insert into chunks (note_id, ord, heading, text, chunk_hash, token_count)
                     values ($1, $2, $3, $4, $5, $6)
                     returning id",
                    &[
                        &note_id,
                        &chunk.ord,
                        &chunk.heading,
                        &chunk.text,
                        &chunk.hash,
                        &token_count,
                    ],
                )
                .await
                .context("写入 chunk 失败")?;
            let chunk_id: i64 = row.get(0);

            if !terms.is_empty() {
                let term_arr: Vec<String> = terms.iter().map(|(t, _)| t.clone()).collect();
                let tf_arr: Vec<i32> = terms.iter().map(|(_, f)| *f).collect();
                tx.execute(
                    "insert into chunk_terms (chunk_id, term, tf)
                     select $1, x.term, x.tf
                     from unnest($2::text[], $3::int[]) as x(term, tf)",
                    &[&chunk_id, &term_arr, &tf_arr],
                )
                .await
                .context("写入 BM25 词项失败")?;
            }

            let lit = vector_literal(vec);
            tx.execute(
                "insert into embeddings (chunk_id, model, dim, vec)
                 values ($1, $2, $3, ($4::text)::vector)",
                &[&chunk_id, &embedder.model_id(), &(vec.len() as i32), &lit],
            )
            .await
            .context("写入 embedding 失败")?;
        }

        tx.commit().await.context("提交事务失败")?;
        stats.notes_written += 1;
        stats.chunks_written += new_idx.len();
    }

    if let Ok(abs) = root.canonicalize() {
        let _ = crate::settings::set(client, "notes_root", &abs.to_string_lossy()).await;
    }
    if let Some(c) = &commit {
        let _ = crate::settings::set(client, "last_commit", c).await;
    }
    Ok(stats)
}

/// 批量导入公开评测语料：每行 `{"id": ..., "text": ...}`。
/// 笔记路径记成 `{prefix}/{id}`，eval 再用同一个 prefix 把命中的块映射回 pid。
pub async fn ingest_jsonl(
    client: &mut Client,
    embedder: &dyn Embedder,
    file: &Path,
    prefix: &str,
    max_chars: usize,
) -> Result<IngestStats> {
    let model_id = embedder.model_id();
    let mut stats = IngestStats::default();
    let f = File::open(file).with_context(|| format!("打开语料失败: {}", file.display()))?;
    let reader = BufReader::new(f);
    let mut batch: Vec<RawRec> = Vec::with_capacity(JSONL_BATCH);

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match parse_record(&line) {
            Ok((id, text)) => {
                stats.files_seen += 1;
                batch.push(RawRec {
                    path: format!("{prefix}/{id}"),
                    text,
                });
            }
            Err(e) => stats.warnings.push(format!("第 {} 行跳过: {e}", idx + 1)),
        }
        if batch.len() >= JSONL_BATCH {
            process_batch(
                client, embedder, &model_id, &mut batch, max_chars, &mut stats,
            )
            .await?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        process_batch(
            client, embedder, &model_id, &mut batch, max_chars, &mut stats,
        )
        .await?;
    }

    Ok(stats)
}

const JSONL_BATCH: usize = 128;

struct RawRec {
    path: String,
    text: String,
}

type LexTerms = Vec<(String, i32)>;

struct Prepared {
    path: String,
    content_hash: String,
    chunks: Vec<(crate::chunk::Chunk, LexTerms, i32)>,
}

fn parse_record(line: &str) -> Result<(String, String)> {
    let v: serde_json::Value = serde_json::from_str(line).context("不是合法 JSON")?;
    let id = match &v["id"] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        _ => bail!("缺少字符串/数字 id"),
    };
    let text = v["text"].as_str().context("缺少 text 字段")?.to_string();
    Ok((id, text))
}

async fn process_batch(
    client: &mut Client,
    embedder: &dyn Embedder,
    model_id: &str,
    batch: &mut [RawRec],
    max_chars: usize,
    stats: &mut IngestStats,
) -> Result<()> {
    let paths: Vec<String> = batch.iter().map(|r| r.path.clone()).collect();
    let rows = client
        .query(
            "select id, path, content_hash from notes where path = any($1::text[])",
            &[&paths],
        )
        .await
        .context("批量查询已有笔记失败")?;
    let mut existing: HashMap<String, (i64, String)> = HashMap::new();
    for r in rows {
        existing.insert(r.get(1), (r.get(0), r.get(2)));
    }

    let note_ids: Vec<i64> = existing.values().map(|(id, _)| *id).collect();
    let mut has_model: HashSet<i64> = HashSet::new();
    if !note_ids.is_empty() {
        let rows = client
            .query(
                "select distinct c.note_id
                 from embeddings e
                 join chunks c on c.id = e.chunk_id
                 where e.model = $1 and c.note_id = any($2::bigint[])",
                &[&model_id, &note_ids],
            )
            .await
            .context("批量检查模型向量失败")?;
        for r in rows {
            has_model.insert(r.get(0));
        }
    }

    let mut prepared: Vec<Prepared> = Vec::new();
    let mut inputs: Vec<String> = Vec::new();

    for rec in batch.iter() {
        let content_hash = sha256_hex(&rec.text);
        if let Some((id, old)) = existing.get(&rec.path)
            && old == &content_hash
            && has_model.contains(id)
        {
            stats.notes_skipped += 1;
            continue;
        }
        let chunks = chunk_markdown(&rec.text, max_chars);
        let mut items = Vec::with_capacity(chunks.len());
        for c in chunks {
            let (terms, token_count) = term_freqs(&c.text);
            inputs.push(c.text.clone());
            items.push((c, terms, token_count));
        }
        prepared.push(Prepared {
            path: rec.path.clone(),
            content_hash,
            chunks: items,
        });
    }

    if prepared.is_empty() {
        return Ok(());
    }

    let vectors = embedder.embed(&inputs).await.context("批量嵌入失败")?;
    if vectors.len() != inputs.len() {
        bail!(
            "嵌入数量 {} 与切块数量 {} 不一致",
            vectors.len(),
            inputs.len()
        );
    }

    let tx = client.transaction().await.context("开启事务失败")?;
    let mut vi = 0usize;

    for p in prepared {
        let row = tx
            .query_one(
                "insert into notes (path, title, content_hash, commit_hash)
                 values ($1, '', $2, null)
                 on conflict (path) do update
                   set content_hash = excluded.content_hash,
                       updated_at = now()
                 returning id",
                &[&p.path, &p.content_hash],
            )
            .await
            .context("写入 note 失败")?;
        let note_id: i64 = row.get(0);

        tx.execute("delete from chunks where note_id = $1", &[&note_id])
            .await
            .context("清理旧 chunk 失败")?;

        for (chunk, terms, token_count) in &p.chunks {
            let row = tx
                .query_one(
                    "insert into chunks (note_id, ord, heading, text, chunk_hash, token_count)
                     values ($1, $2, $3, $4, $5, $6)
                     returning id",
                    &[
                        &note_id,
                        &chunk.ord,
                        &chunk.heading,
                        &chunk.text,
                        &chunk.hash,
                        token_count,
                    ],
                )
                .await
                .context("写入 chunk 失败")?;
            let chunk_id: i64 = row.get(0);

            if !terms.is_empty() {
                let term_arr: Vec<String> = terms.iter().map(|(t, _)| t.clone()).collect();
                let tf_arr: Vec<i32> = terms.iter().map(|(_, f)| *f).collect();
                tx.execute(
                    "insert into chunk_terms (chunk_id, term, tf)
                     select $1, x.term, x.tf
                     from unnest($2::text[], $3::int[]) as x(term, tf)",
                    &[&chunk_id, &term_arr, &tf_arr],
                )
                .await
                .context("写入 BM25 词项失败")?;
            }

            let vec = &vectors[vi];
            vi += 1;
            let lit = vector_literal(vec);
            tx.execute(
                "insert into embeddings (chunk_id, model, dim, vec)
                 values ($1, $2, $3, ($4::text)::vector)",
                &[&chunk_id, &model_id, &(vec.len() as i32), &lit],
            )
            .await
            .context("写入 embedding 失败")?;
        }

        stats.notes_written += 1;
        stats.chunks_written += p.chunks.len();
    }

    tx.commit().await.context("提交事务失败")?;
    Ok(())
}

fn collect_markdown(root: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !(e.depth() > 0 && name.starts_with('.'))
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x.eq_ignore_ascii_case("md"))
                .unwrap_or(false)
        })
        .map(|e| e.into_path())
        .collect();
    files.sort();
    files
}

/// 零依赖拿 HEAD：索引是可重建的派生数据，commit 只是 provenance 标注。
pub fn git_head(repo: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}
