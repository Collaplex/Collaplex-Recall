use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::Value;
use tokio_postgres::Client;

use crate::embed::Embedder;
use crate::rerank::{self, RerankConfig};
use crate::search::{Fusion, SearchOptions, hybrid_search};

/// 标准 IR 输入：queries.jsonl 是 `{"id","text"}`，
/// qrels.jsonl 是 `{"qid","pid","score"}`（score>0 才算相关）。
pub struct EvalOptions {
    pub k: usize,
    pub candidates: usize,
    pub use_vector: bool,
    pub use_lexical: bool,
    pub fusion: Fusion,
    /// 是否用交叉编码器精排
    pub use_rerank: bool,
    /// 入库时用的路径前缀，用来把 chunk 映射回原始 pid。
    pub prefix: String,
}

#[derive(Debug, Serialize)]
pub struct EvalReport {
    pub queries_total: usize,
    pub queries_scored: usize,
    pub k: usize,
    pub recall_at_k: f64,
    pub hit_at_k: f64,
    pub mrr_at_k: f64,
    pub ndcg_at_k: f64,
    pub queries_without_relevant_hit: usize,
    pub queries_without_any_hit: usize,
    pub warnings: Vec<String>,
}

pub async fn run_eval(
    client: &Client,
    embedder: Option<&dyn Embedder>,
    queries_path: &Path,
    qrels_path: &Path,
    opts: &EvalOptions,
) -> Result<EvalReport> {
    let queries = read_queries(queries_path)?;
    let qrels = read_qrels(qrels_path)?;
    let path_prefix = if opts.prefix.trim().is_empty() {
        String::new()
    } else {
        format!("{}/", opts.prefix.trim_end_matches('/'))
    };
    let search_opts = SearchOptions {
        k: opts.k,
        candidates: opts.candidates,
        use_vector: opts.use_vector,
        use_lexical: opts.use_lexical,
        fusion: opts.fusion,
    };

    let mut report = EvalReport {
        queries_total: queries.len(),
        queries_scored: 0,
        k: opts.k,
        recall_at_k: 0.0,
        hit_at_k: 0.0,
        mrr_at_k: 0.0,
        ndcg_at_k: 0.0,
        queries_without_relevant_hit: 0,
        queries_without_any_hit: 0,
        warnings: Vec::new(),
    };

    for (qid, text) in &queries {
        let Some(gold) = qrels.get(qid) else {
            continue;
        };
        if gold.is_empty() {
            continue;
        }

        let mut outcome = hybrid_search(client, embedder, text, &search_opts).await?;
        if opts.use_rerank && !outcome.hits.is_empty() {
            let cfg = RerankConfig::from_env();
            let docs: Vec<String> = outcome
                .hits
                .iter()
                .map(|h| {
                    let t: String = h.text.chars().take(400).collect();
                    format!("{}\n{}", h.heading, t)
                })
                .collect();
            match rerank::rerank(&cfg, text, &docs).await {
                Ok((scores, _)) => {
                    let mut old = std::mem::take(&mut outcome.hits);
                    let mut new_hits = Vec::with_capacity(old.len());
                    for (idx, _) in &scores {
                        if let Some(h) = old.get(*idx) {
                            new_hits.push(h.clone());
                        }
                    }
                    for h in old.drain(..) {
                        if !new_hits.iter().any(|x| x.chunk_id == h.chunk_id) {
                            new_hits.push(h);
                        }
                    }
                    outcome.hits = new_hits;
                }
                Err(e) => report
                    .warnings
                    .push(format!("精排失败（保留原顺序）: {e:#}")),
            }
        }
        if outcome.hits.is_empty() {
            report.queries_without_any_hit += 1;
        }
        for w in &outcome.warnings {
            if !report.warnings.contains(w) {
                report.warnings.push(w.clone());
            }
        }

        // 同一篇文档可能有多个 chunk 命中，评测时按 pid 去重保名次。
        let mut ranked: Vec<&str> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for h in &outcome.hits {
            if let Some(pid) = h.path.strip_prefix(&path_prefix)
                && seen.insert(pid)
            {
                ranked.push(pid);
            }
        }

        let mut rel_positions = Vec::new();
        for (i, pid) in ranked.iter().enumerate() {
            if gold.contains(*pid) {
                rel_positions.push(i);
            }
        }

        let mut dcg = 0.0;
        for &i in &rel_positions {
            dcg += 1.0 / ((i as f64 + 2.0).log2());
        }
        let idcg: f64 = (0..gold.len().min(opts.k))
            .map(|j| 1.0 / (j as f64 + 2.0).log2())
            .sum();

        report.queries_scored += 1;
        report.recall_at_k += rel_positions.len().min(gold.len()) as f64 / gold.len() as f64;
        report.ndcg_at_k += if idcg > 0.0 { dcg / idcg } else { 0.0 };
        if let Some(&first) = rel_positions.first() {
            report.hit_at_k += 1.0;
            report.mrr_at_k += 1.0 / (first as f64 + 1.0);
        } else {
            report.queries_without_relevant_hit += 1;
        }
    }

    let n = report.queries_scored.max(1) as f64;
    report.recall_at_k /= n;
    report.hit_at_k /= n;
    report.mrr_at_k /= n;
    report.ndcg_at_k /= n;

    Ok(report)
}

fn read_queries(path: &Path) -> Result<Vec<(String, String)>> {
    let f = File::open(path).with_context(|| format!("打开 queries 失败: {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .with_context(|| format!("queries 第 {} 行不是 JSON", i + 1))?;
        out.push((json_id(&v, "id")?, json_text(&v, "text")?));
    }
    Ok(out)
}

fn read_qrels(path: &Path) -> Result<HashMap<String, HashSet<String>>> {
    let f = File::open(path).with_context(|| format!("打开 qrels 失败: {}", path.display()))?;
    let mut out: HashMap<String, HashSet<String>> = HashMap::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .with_context(|| format!("qrels 第 {} 行不是 JSON", i + 1))?;
        let score = v["score"]
            .as_f64()
            .or_else(|| v["score"].as_str().and_then(|s| s.parse().ok()))
            .unwrap_or(1.0);
        if score <= 0.0 {
            continue;
        }
        out.entry(json_id(&v, "qid")?)
            .or_default()
            .insert(json_id(&v, "pid")?);
    }
    Ok(out)
}

fn json_id(v: &Value, key: &str) -> Result<String> {
    match &v[key] {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        _ => bail!("缺少字符串/数字字段 {key}"),
    }
}

fn json_text(v: &Value, key: &str) -> Result<String> {
    v[key]
        .as_str()
        .map(|s| s.to_string())
        .with_context(|| format!("缺少字符串字段 {key}"))
}
