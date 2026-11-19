use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use tokio_postgres::Client;

use crate::embed::{Embedder, vector_literal};

const RRF_K: f64 = 60.0;

/// 两路结果怎么合：
/// - RRF：并行召回，按名次加权融合（字面弱，会稀释向量）。
/// - Rerank：两段式——向量先召回候选，候选集内用 BM25 重排，不引入新文档。
#[derive(Debug, Clone, Copy)]
pub enum Fusion {
    Rrf {
        vector_weight: f64,
        lexical_weight: f64,
    },
    Rerank {
        alpha: f64,
    },
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub k: usize,
    pub candidates: usize,
    pub use_vector: bool,
    pub use_lexical: bool,
    pub fusion: Fusion,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            k: 10,
            candidates: 50,
            use_vector: true,
            use_lexical: true,
            fusion: Fusion::Rerank { alpha: 0.05 },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CandidateCounts {
    pub vector: usize,
    pub lexical: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub rank: usize,
    pub chunk_id: i64,
    pub path: String,
    pub heading: String,
    pub text: String,
    /// Reciprocal Rank Fusion 分数；只用来排序，不是相似度。
    pub rrf: f64,
    pub vector_rank: Option<usize>,
    pub vector_score: Option<f64>,
    pub lexical_rank: Option<usize>,
    pub lexical_score: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct SearchOutcome {
    pub query: String,
    pub embedder: String,
    pub k: usize,
    pub candidates: CandidateCounts,
    pub elapsed_ms: u128,
    pub warnings: Vec<String>,
    pub hits: Vec<Hit>,
}

#[derive(Default)]
struct Fused {
    rrf: f64,
    vector_rank: Option<usize>,
    vector_score: Option<f64>,
    lexical_rank: Option<usize>,
    lexical_score: Option<f64>,
}

const VECTOR_SQL: &str = r#"
select e.chunk_id, (1 - (e.vec <=> ($1::text)::vector))::float8 as score
from embeddings e
where e.model = $2
order by e.vec <=> ($1::text)::vector
limit $3
"#;

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// BM25 放在 Rust 里算：每个词项一条倒排索引查询，再加权累加。
/// 不在 SQL 里 join 是因为 3000 万行倒排表的基数估计不稳，
/// 规划器会退化成全表扫描（实测 1 秒/查询）；逐词项取倒排稳定在几十毫秒。
async fn lexical_search(
    client: &Client,
    query_terms: &[String],
    limit: usize,
) -> Result<Vec<(i64, f64)>> {
    if query_terms.is_empty() {
        return Ok(Vec::new());
    }
    let row = client
        .query_one(
            "select count(*), coalesce(avg(token_count), 0)::float8 from chunks",
            &[],
        )
        .await
        .context("读取 BM25 语料统计失败")?;
    let n: i64 = row.get(0);
    let avgdl: f64 = row.get(1);
    if n == 0 {
        return Ok(Vec::new());
    }
    let n = n as f64;
    let avgdl = avgdl.max(1.0);

    let mut scores: HashMap<i64, f64> = HashMap::new();
    for term in query_terms {
        let rows = client
            .query(
                "select ct.chunk_id, ct.tf, c.token_count
                 from chunk_terms ct
                 join chunks c on c.id = ct.chunk_id
                 where ct.term = $1",
                &[term],
            )
            .await
            .with_context(|| format!("读取 BM25 倒排失败: {term}"))?;
        let df = rows.len() as f64;
        if df == 0.0 {
            continue;
        }
        let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
        for r in rows {
            let chunk_id: i64 = r.get(0);
            let tf: i32 = r.get(1);
            let dl: i32 = r.get(2);
            let tf = tf as f64;
            let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl as f64 / avgdl);
            if denom > 0.0 {
                *scores.entry(chunk_id).or_insert(0.0) += idf * tf * (BM25_K1 + 1.0) / denom;
            }
        }
    }

    let mut ranked: Vec<(i64, f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(limit);
    Ok(ranked)
}

/// 只给向量候选集算 BM25：取查询词项的 df、候选块的 tf 和长度即可。
async fn bm25_for_candidates(
    client: &Client,
    query_terms: &[String],
    ids: &[i64],
) -> Result<HashMap<i64, f64>> {
    let row = client
        .query_one(
            "select count(*), coalesce(avg(token_count), 0)::float8 from chunks",
            &[],
        )
        .await
        .context("读取 BM25 语料统计失败")?;
    let n: i64 = row.get(0);
    let avgdl: f64 = row.get(1);
    if n == 0 || ids.is_empty() {
        return Ok(HashMap::new());
    }
    let n = n as f64;
    let avgdl = avgdl.max(1.0);

    let df_rows = client
        .query(
            "select term, count(*)::float8
             from chunk_terms
             where term = any($1::text[])
             group by term",
            &[&query_terms],
        )
        .await
        .context("读取 BM25 词项 df 失败")?;
    let mut df: HashMap<String, f64> = HashMap::new();
    for r in df_rows {
        df.insert(r.get(0), r.get(1));
    }

    let rows = client
        .query(
            "select ct.chunk_id, ct.term, ct.tf, c.token_count
             from chunk_terms ct
             join chunks c on c.id = ct.chunk_id
             where ct.chunk_id = any($1::bigint[]) and ct.term = any($2::text[])",
            &[&ids, &query_terms],
        )
        .await
        .context("读取候选块 BM25 词频失败")?;

    let mut scores: HashMap<i64, f64> = HashMap::new();
    for r in rows {
        let chunk_id: i64 = r.get(0);
        let term: String = r.get(1);
        let tf: i32 = r.get(2);
        let dl: i32 = r.get(3);
        let Some(&d) = df.get(&term) else {
            continue;
        };
        let idf = (1.0 + (n - d + 0.5) / (d + 0.5)).ln();
        let tf = tf as f64;
        let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl as f64 / avgdl);
        if denom > 0.0 {
            *scores.entry(chunk_id).or_insert(0.0) += idf * tf * (BM25_K1 + 1.0) / denom;
        }
    }
    Ok(scores)
}

fn ranked_vector_only(vector_rows: &[(i64, f64)]) -> Vec<(i64, Fused)> {
    vector_rows
        .iter()
        .enumerate()
        .map(|(rank, (id, score))| {
            (
                *id,
                Fused {
                    rrf: *score,
                    vector_rank: Some(rank + 1),
                    vector_score: Some(*score),
                    ..Default::default()
                },
            )
        })
        .collect()
}

fn ranked_rerank(
    vector_rows: &[(i64, f64)],
    bm25: &HashMap<i64, f64>,
    alpha: f64,
) -> Vec<(i64, Fused)> {
    let max_bm = bm25.values().cloned().fold(0.0, f64::max);
    let mut bm_sorted: Vec<(i64, f64)> = bm25.iter().map(|(id, s)| (*id, *s)).collect();
    bm_sorted.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let lex_rank: HashMap<i64, usize> = bm_sorted
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (*id, i + 1))
        .collect();

    vector_rows
        .iter()
        .enumerate()
        .map(|(rank, (id, cos))| {
            let raw = bm25.get(id).copied().unwrap_or(0.0);
            let norm = if max_bm > 0.0 { raw / max_bm } else { 0.0 };
            (
                *id,
                Fused {
                    rrf: cos + alpha * norm,
                    vector_rank: Some(rank + 1),
                    vector_score: Some(*cos),
                    lexical_rank: lex_rank.get(id).copied(),
                    lexical_score: bm25.get(id).copied(),
                },
            )
        })
        .collect()
}

fn fuse(fused: &mut HashMap<i64, Fused>, rows: &[(i64, f64)], is_vector: bool, weight: f64) {
    for (rank, (id, score)) in rows.iter().enumerate() {
        let entry = fused.entry(*id).or_default();
        entry.rrf += weight / (RRF_K + (rank + 1) as f64);
        if is_vector {
            entry.vector_rank = Some(rank + 1);
            entry.vector_score = Some(*score);
        } else {
            entry.lexical_rank = Some(rank + 1);
            entry.lexical_score = Some(*score);
        }
    }
}

pub async fn hybrid_search(
    client: &Client,
    embedder: Option<&dyn Embedder>,
    query: &str,
    opts: &SearchOptions,
) -> Result<SearchOutcome> {
    let started = Instant::now();
    let mut warnings = Vec::new();
    let mut fused: HashMap<i64, Fused> = HashMap::new();
    let mut vector_rows: Vec<(i64, f64)> = Vec::new();
    let mut lexical_rows: Vec<(i64, f64)> = Vec::new();
    let mut embedder_id = "none".to_string();

    if opts.use_vector {
        match embedder {
            Some(e) => {
                embedder_id = e.model_id();
                match e.embed_query(query).await {
                    Ok(qvec) => {
                        let q = vector_literal(&qvec);
                        let limit = opts.candidates as i64;
                        match client.query(VECTOR_SQL, &[&q, &embedder_id, &limit]).await {
                            Ok(rows) => {
                                vector_rows = rows
                                    .iter()
                                    .map(|r| (r.get::<_, i64>(0), r.get::<_, f64>(1)))
                                    .collect();
                            }
                            Err(err) => {
                                warnings.push(format!("向量检索失败（已降级为纯字面检索）: {err}"))
                            }
                        }
                    }
                    Err(err) => warnings.push(format!("嵌入失败（已降级为纯字面检索）: {err:#}")),
                }
            }
            None => warnings.push("没有嵌入器，跳过向量检索".to_string()),
        }
    }

    // 前缀不同会算出不同 model_id，结果就是“向量 0 候选”却不报错。
    // 这里给个明确提示，免得看起来像检索质量突然变差。
    if opts.use_vector
        && vector_rows.is_empty()
        && embedder_id != "none"
        && let Ok(rows) = client
            .query(
                "select model, count(*) from embeddings group by model order by 1",
                &[],
            )
            .await
        && !rows.is_empty()
    {
        let list = rows
            .iter()
            .take(3)
            .map(|r| format!("{}:{}", r.get::<_, String>(0), r.get::<_, i64>(1)))
            .collect::<Vec<_>>()
            .join(", ");
        warnings.push(format!(
            "向量 0 候选：当前模型 {embedder_id} 在库里没有向量（已有 {list}）"
        ));
    }

    let mut qterms = crate::token::terms(query);
    qterms.sort();
    qterms.dedup();
    qterms.truncate(96);

    let mut ranked: Vec<(i64, Fused)> = match opts.fusion {
        Fusion::Rerank { alpha } => {
            if vector_rows.is_empty() {
                if opts.use_lexical && !qterms.is_empty() {
                    match lexical_search(client, &qterms, opts.candidates).await {
                        Ok(rows) => {
                            lexical_rows = rows;
                            warnings.push("向量无结果，退化为纯字面检索".to_string());
                        }
                        Err(err) => warnings.push(format!("字面检索失败: {err}")),
                    }
                }
                let mut fallback = HashMap::new();
                fuse(&mut fallback, &lexical_rows, false, 1.0);
                fallback.into_iter().collect()
            } else if opts.use_lexical && !qterms.is_empty() {
                let ids: Vec<i64> = vector_rows.iter().map(|(id, _)| *id).collect();
                match bm25_for_candidates(client, &qterms, &ids).await {
                    Ok(scores) => ranked_rerank(&vector_rows, &scores, alpha),
                    Err(err) => {
                        warnings.push(format!("BM25 重排失败，保留向量排序: {err}"));
                        ranked_vector_only(&vector_rows)
                    }
                }
            } else {
                ranked_vector_only(&vector_rows)
            }
        }
        Fusion::Rrf {
            vector_weight,
            lexical_weight,
        } => {
            if opts.use_lexical && !qterms.is_empty() {
                match lexical_search(client, &qterms, opts.candidates).await {
                    Ok(rows) => lexical_rows = rows,
                    Err(err) => warnings.push(format!("字面检索失败: {err}")),
                }
            }
            fuse(&mut fused, &vector_rows, true, vector_weight);
            fuse(&mut fused, &lexical_rows, false, lexical_weight);
            fused.into_iter().collect()
        }
    };
    ranked.sort_by(|(a_id, a), (b_id, b)| {
        b.rrf
            .partial_cmp(&a.rrf)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a_id.cmp(b_id))
    });
    ranked.truncate(opts.k);

    let hits = hydrate(client, &ranked).await?;

    Ok(SearchOutcome {
        query: query.to_string(),
        embedder: embedder_id,
        k: opts.k,
        candidates: CandidateCounts {
            vector: vector_rows.len(),
            lexical: lexical_rows.len(),
        },
        elapsed_ms: started.elapsed().as_millis(),
        warnings,
        hits,
    })
}

async fn hydrate(client: &Client, ranked: &[(i64, Fused)]) -> Result<Vec<Hit>> {
    if ranked.is_empty() {
        return Ok(Vec::new());
    }
    let ids: Vec<i64> = ranked.iter().map(|(id, _)| *id).collect();
    let rows = client
        .query(
            "select c.id, n.path, c.heading, c.text
             from chunks c
             join notes n on n.id = c.note_id
             where c.id = any($1::bigint[])",
            &[&ids],
        )
        .await
        .context("取回命中文档失败")?;

    let mut by_id: HashMap<i64, (String, String, String)> = HashMap::new();
    for r in rows {
        by_id.insert(
            r.get::<_, i64>(0),
            (
                r.get::<_, String>(1),
                r.get::<_, String>(2),
                r.get::<_, String>(3),
            ),
        );
    }

    let mut hits = Vec::with_capacity(ranked.len());
    for (rank, (id, f)) in ranked.iter().enumerate() {
        if let Some((path, heading, text)) = by_id.remove(id) {
            hits.push(Hit {
                rank: rank + 1,
                chunk_id: *id,
                path,
                heading,
                text,
                rrf: f.rrf,
                vector_rank: f.vector_rank,
                vector_score: f.vector_score,
                lexical_rank: f.lexical_rank,
                lexical_score: f.lexical_score,
            });
        }
    }
    Ok(hits)
}
