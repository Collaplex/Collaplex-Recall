use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use tokio_postgres::Client;

use crate::chunk::sha256_hex;
use crate::token;

#[derive(Debug, Clone, Serialize)]
pub struct MemoryRow {
    pub id: i64,
    pub text: String,
    pub note_path: Option<String>,
    pub status: String,
    pub pinned: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryHit {
    pub id: i64,
    pub text: String,
    pub note_path: Option<String>,
    /// 图一跳扩展进来的邻居，记录来源种子；直接命中为 None。
    pub via: Option<i64>,
}

pub enum AddOutcome {
    Created(i64),
    Existing(i64),
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub fn edge_eta() -> f64 {
    env_f64("NOTE_EDGE_ETA", 0.1)
}

fn edge_half_life_days() -> f64 {
    env_f64("NOTE_EDGE_HALF_LIFE_DAYS", 30.0)
}

pub fn edge_min() -> f64 {
    env_f64("NOTE_EDGE_MIN", 0.15)
}

/// 同一轮被共同使用的记忆之间加强边（Hebbian，只作用于记忆库）。
pub async fn strengthen(client: &Client, ids: &[i64]) -> Result<()> {
    let eta = edge_eta();
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            if ids[i] == ids[j] {
                continue;
            }
            let (a, b) = if ids[i] < ids[j] {
                (ids[i], ids[j])
            } else {
                (ids[j], ids[i])
            };
            client
                .execute(
                    "insert into memory_edges (a_id, b_id, weight, t_updated)
                     values ($1, $2, $3, now())
                     on conflict (a_id, b_id) do update
                       set weight = least(memory_edges.weight + $3, 1.0),
                           t_updated = now()",
                    &[&a, &b, &eta],
                )
                .await
                .context("写入记忆图边失败")?;
        }
    }
    Ok(())
}

/// 负反馈：对一轮里用到的记忆做 LTD（削弱相关边）。
pub async fn penalize(client: &Client, ids: &[i64]) -> Result<()> {
    let eta = edge_eta();
    for id in ids {
        weaken_edges(client, *id, eta).await?;
    }
    Ok(())
}

/// 失效记忆的确定性重提炼：当前笔记里仍有高度相似的 chunk 就重新绑定并恢复；
/// 否则保持 stale。模型重写版本暂不做。
pub async fn reextract(client: &Client, limit: usize) -> Result<(usize, usize)> {
    let rows = client
        .query(
            "select m.id, m.text, m.note_id, n.content_hash, n.commit_hash
             from memories m
             join notes n on n.id = m.note_id
             where m.status = 'stale' and not m.pinned
             order by m.id
             limit $1",
            &[&(limit as i64)],
        )
        .await
        .context("读取失效记忆失败")?;

    let mut revived = 0usize;
    let mut left = 0usize;
    for r in rows {
        let id: i64 = r.get(0);
        let text: String = r.get(1);
        let note_id: i64 = r.get(2);
        let note_hash: String = r.get(3);
        let commit: Option<String> = r.get(4);

        let chunks = client
            .query(
                "select id, chunk_hash, text from chunks where note_id = $1",
                &[&note_id],
            )
            .await
            .context("读取当前 chunk 失败")?;
        if chunks.is_empty() {
            left += 1;
            continue;
        }
        let mterms: HashSet<String> = token::terms(&text).into_iter().collect();
        let mut best: Option<(f64, usize, i64, String)> = None;
        for c in chunks {
            let cid: i64 = c.get(0);
            let hash: String = c.get(1);
            let ctext: String = c.get(2);
            let cterms: HashSet<String> = token::terms(&ctext).into_iter().collect();
            // 短记忆对长 chunk：看记忆自身有多少词项被 chunk 覆盖。
            let inter = mterms.intersection(&cterms).count();
            let score = if mterms.is_empty() {
                0.0
            } else {
                inter as f64 / mterms.len() as f64
            };
            if best.as_ref().map(|b| score > b.0).unwrap_or(true) {
                best = Some((score, inter, cid, hash));
            }
        }
        match best {
            Some((score, inter, cid, hash)) if inter >= 3 && score >= 0.25 => {
                client
                    .execute(
                        "update memories
                         set status = 'active', chunk_id = $1, chunk_hash = $2,
                             note_content_hash = $3, commit_hash = $4, updated_at = now()
                         where id = $5",
                        &[&cid, &hash, &note_hash, &commit, &id],
                    )
                    .await
                    .context("恢复记忆失败")?;
                revived += 1;
            }
            _ => left += 1,
        }
    }
    Ok((revived, left))
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeSensitivity {
    pub a_id: i64,
    pub b_id: i64,
    pub weight: f64,
    pub decayed: f64,
    /// 与一跳阈值的最小距离；越小越敏感。
    pub margin: f64,
    /// ±delta 内是否会跨过阈值。
    pub flips: bool,
}

/// 有限差分：只看边权扰动会不会改变“是否参与一跳扩展”，不碰模型。
pub async fn edge_sensitivity(client: &Client, delta: f64) -> Result<Vec<EdgeSensitivity>> {
    let half_life = edge_half_life_days().max(1.0);
    let min = edge_min();
    let rows = client
        .query(
            "select a_id, b_id, weight,
                    (extract(epoch from (now() - t_updated)) / 86400.0)::float8 as days
             from memory_edges",
            &[],
        )
        .await
        .context("读取记忆图边失败")?;
    let mut out: Vec<EdgeSensitivity> = rows
        .iter()
        .map(|r| {
            let a_id: i64 = r.get(0);
            let b_id: i64 = r.get(1);
            let weight: f64 = r.get(2);
            let days: f64 = r.get(3);
            let decayed = weight * 0.5f64.powf(days / half_life);
            let margin = (decayed - min).abs();
            EdgeSensitivity {
                a_id,
                b_id,
                weight,
                decayed,
                margin,
                flips: margin <= delta,
            }
        })
        .collect();
    out.sort_by(|a, b| {
        a.margin
            .partial_cmp(&b.margin)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

/// 离线 GC：删掉衰减后低于阈值、且长期没更新过的边。
pub async fn gc_edges(client: &Client, min_weight: f64, older_days: i64) -> Result<u64> {
    let half_life = edge_half_life_days().max(1.0);
    let n = client
        .execute(
            "delete from memory_edges
             where t_updated < now() - ($2::int * interval '1 day')
               and weight * power(
                     0.5::float8,
                     (extract(epoch from (now() - t_updated)) / 86400.0)::float8 / $3
                   ) < $1",
            &[&min_weight, &(older_days as i32), &half_life],
        )
        .await
        .context("清理记忆图边失败")?;
    Ok(n)
}

/// LTD：记忆失效/删除时削弱它相关的所有边。
async fn weaken_edges(client: &Client, id: i64, eta: f64) -> Result<()> {
    client
        .execute(
            "update memory_edges
             set weight = greatest(weight - $1, 0), t_updated = now()
             where a_id = $2 or b_id = $2",
            &[&eta, &id],
        )
        .await
        .context("削弱记忆图边失败")?;
    Ok(())
}

/// 包含系数：短文本的词项有多大比例被长文本覆盖。
/// 比 Jaccard 更适合“同一事实、一个写得长一点”的去重。
fn containment(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f64;
    inter / a.len().min(b.len()) as f64
}

/// 新增记忆条：必须绑定一篇已有笔记，provenance 取当时的 content_hash/commit。
pub async fn add(
    client: &Client,
    text: &str,
    note_path: &str,
    chunk_id: Option<i64>,
    pinned: bool,
) -> Result<AddOutcome> {
    let text = text.trim();
    if text.is_empty() {
        bail!("记忆内容不能为空");
    }
    let note = client
        .query_opt(
            "select id, content_hash, commit_hash from notes where path = $1",
            &[&note_path],
        )
        .await
        .context("查询来源笔记失败")?;
    let Some(note) = note else {
        bail!("找不到来源笔记：{note_path}");
    };
    let note_id: i64 = note.get(0);
    let content_hash: String = note.get(1);
    let commit_hash: Option<String> = note.get(2);
    let text_hash = sha256_hex(text);

    // chunk 级 provenance：只有 chunk 确实属于这篇笔记时才绑定。
    let (chunk_id, chunk_hash) = match chunk_id {
        Some(cid) => match client
            .query_opt(
                "select chunk_hash from chunks where id = $1 and note_id = $2",
                &[&cid, &note_id],
            )
            .await
            .context("查询来源 chunk 失败")?
        {
            Some(r) => (Some(cid), Some(r.get::<_, String>(0))),
            None => (None, None),
        },
        None => (None, None),
    };

    let existing = client
        .query_opt(
            "select id from memories
             where text_hash = $1 and note_id = $2 and status <> 'deleted'",
            &[&text_hash, &note_id],
        )
        .await
        .context("检查重复记忆失败")?;
    if let Some(row) = existing {
        return Ok(AddOutcome::Existing(row.get(0)));
    }

    // 近义去重：同一来源下，短文本词项被覆盖 ≥ 0.8 视为同一条事实。
    let new_terms: HashSet<String> = token::terms(text).into_iter().collect();
    let rows = client
        .query(
            "select id, text from memories where note_id = $1 and status = 'active'",
            &[&note_id],
        )
        .await
        .context("读取已有关键记忆失败")?;
    for row in rows {
        let old: String = row.get(1);
        let old_terms: HashSet<String> = token::terms(&old).into_iter().collect();
        if containment(&new_terms, &old_terms) >= 0.7 {
            return Ok(AddOutcome::Existing(row.get(0)));
        }
    }

    let row = client
        .query_one(
            "insert into memories
               (text, text_hash, note_id, note_path, commit_hash, note_content_hash,
                chunk_id, chunk_hash, pinned)
             values ($1, $2, $3, $4, $5, $6, $7, $8, $9)
             returning id",
            &[
                &text,
                &text_hash,
                &note_id,
                &note_path,
                &commit_hash,
                &content_hash,
                &chunk_id,
                &chunk_hash,
                &pinned,
            ],
        )
        .await
        .context("写入记忆失败")?;
    Ok(AddOutcome::Created(row.get(0)))
}

/// 读取前刷新：来源笔记内容变了或没了，就把未固定记忆降级为 stale（不删）。
pub async fn refresh(client: &Client) -> Result<u64> {
    let eta = edge_eta();
    let changed = client
        .query(
            "update memories m
             set status = 'stale', updated_at = now()
             from notes n
             where m.note_id = n.id
               and m.status = 'active'
               and not m.pinned
               and m.note_content_hash is distinct from n.content_hash
             returning m.id",
            &[],
        )
        .await
        .context("检查记忆 provenance 失败")?;
    let missing = client
        .query(
            "update memories m
             set status = 'stale', updated_at = now()
             where m.status = 'active'
               and not m.pinned
               and m.note_id is not null
               and not exists (select 1 from notes n where n.id = m.note_id)
             returning m.id",
            &[],
        )
        .await
        .context("检查失效记忆失败")?;
    let chunk_changed = client
        .query(
            "update memories m
             set status = 'stale', updated_at = now()
             from chunks c
             where m.chunk_id = c.id
               and m.status = 'active'
               and not m.pinned
               and m.chunk_hash is distinct from c.chunk_hash
             returning m.id",
            &[],
        )
        .await
        .context("检查 chunk provenance 失败")?;
    let chunk_missing = client
        .query(
            "update memories m
             set status = 'stale', updated_at = now()
             where m.status = 'active'
               and not m.pinned
               and m.chunk_id is not null
               and not exists (select 1 from chunks c where c.id = m.chunk_id)
             returning m.id",
            &[],
        )
        .await
        .context("检查失效 chunk 失败")?;

    let mut total = 0usize;
    for row in changed
        .iter()
        .chain(missing.iter())
        .chain(chunk_changed.iter())
        .chain(chunk_missing.iter())
    {
        weaken_edges(client, row.get(0), eta).await?;
        total += 1;
    }
    Ok(total as u64)
}

/// 列出未删除的记忆（含 stale），active 在前。
pub async fn list(client: &Client) -> Result<Vec<MemoryRow>> {
    refresh(client).await?;
    let rows = client
        .query(
            "select id, text, note_path, status, pinned
             from memories
             where status <> 'deleted'
             order by (status = 'stale'), id",
            &[],
        )
        .await
        .context("读取记忆失败")?;
    Ok(rows
        .iter()
        .map(|r| MemoryRow {
            id: r.get(0),
            text: r.get(1),
            note_path: r.get(2),
            status: r.get(3),
            pinned: r.get(4),
        })
        .collect())
}

/// 软删除：只改状态，不物理删除。
pub async fn remove(client: &Client, id: i64) -> Result<bool> {
    let row = client
        .query_opt(
            "update memories
             set status = 'deleted', deleted_at = now(), updated_at = now()
             where id = $1 and status <> 'deleted'
             returning id",
            &[&id],
        )
        .await
        .context("删除记忆失败")?;
    if row.is_some() {
        weaken_edges(client, id, edge_eta()).await?;
        let _ = crate::feedback::record(
            client,
            None,
            None,
            "delete_memory",
            Some(&id.to_string()),
            serde_json::json!({"memory_id": id}),
        )
        .await;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// 确定性记忆检索：词项重合取种子，再做一跳图扩展（weight × 时间衰减）。
pub async fn retrieve(client: &Client, query: &str, limit: usize) -> Result<Vec<MemoryHit>> {
    refresh(client).await?;
    let qterms: HashSet<String> = token::terms(query).into_iter().collect();
    if qterms.is_empty() {
        return Ok(Vec::new());
    }

    let rows = client
        .query(
            "select id, text, note_path from memories where status = 'active'",
            &[],
        )
        .await
        .context("读取记忆失败")?;

    let mut scored: Vec<(i64, i64, MemoryHit)> = Vec::new();
    for r in rows {
        let id: i64 = r.get(0);
        let text: String = r.get(1);
        let score = token::terms(&text)
            .iter()
            .filter(|t| qterms.contains(*t))
            .count();
        if score > 0 {
            scored.push((
                score as i64,
                id,
                MemoryHit {
                    id,
                    text,
                    note_path: r.get(2),
                    via: None,
                },
            ));
        }
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut out: Vec<MemoryHit> = scored
        .into_iter()
        .take(limit)
        .map(|(_, _, hit)| hit)
        .collect();
    if out.len() >= limit || out.is_empty() {
        return Ok(out);
    }

    let seed_ids: Vec<i64> = out.iter().map(|m| m.id).collect();
    let edge_rows = client
        .query(
            "select case when a_id = any($1::bigint[]) then a_id else b_id end as seed,
                    case when a_id = any($1::bigint[]) then b_id else a_id end as other,
                    weight,
                    (extract(epoch from (now() - t_updated)) / 86400.0)::float8 as days
             from memory_edges
             where a_id = any($1::bigint[]) or b_id = any($1::bigint[])",
            &[&seed_ids],
        )
        .await
        .context("读取记忆图边失败")?;

    let half_life = edge_half_life_days().max(1.0);
    let mut neigh: Vec<(f64, i64, i64)> = Vec::new();
    for r in edge_rows {
        let seed: i64 = r.get(0);
        let other: i64 = r.get(1);
        if out.iter().any(|m| m.id == other) {
            continue;
        }
        let weight: f64 = r.get(2);
        let days: f64 = r.get(3);
        let decayed = weight * 0.5f64.powf(days / half_life);
        if decayed >= edge_min() {
            neigh.push((decayed, seed, other));
        }
    }
    neigh.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });

    for (_, seed, other) in neigh {
        if out.len() >= limit {
            break;
        }
        if out.iter().any(|m| m.id == other) {
            continue;
        }
        if let Some(row) = client
            .query_opt(
                "select id, text, note_path from memories where id = $1 and status = 'active'",
                &[&other],
            )
            .await
            .context("读取图邻居失败")?
        {
            out.push(MemoryHit {
                id: row.get(0),
                text: row.get(1),
                note_path: row.get(2),
                via: Some(seed),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(s: &str) -> HashSet<String> {
        token::terms(s).into_iter().collect()
    }

    #[test]
    fn containment_catches_reworded_fact() {
        let a = terms("等响度补偿在200Hz做负8dB的搁架");
        let b = terms("等响度补偿在200Hz做负8dB的搁架，逐样本验证通过。");
        let c = terms("等响度补偿与AGC配合使用，避免放大环境底噪。");
        assert!(containment(&a, &b) >= 0.8);
        assert!(containment(&a, &c) < 0.8);
    }
}
