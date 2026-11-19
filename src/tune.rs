use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use tokio_postgres::Client;

#[derive(Debug, Serialize)]
pub struct TuneReport {
    pub turns_with_feedback: usize,
    pub positive_events: usize,
    pub negative_events: usize,
    pub best_alpha: f64,
    pub best_score: f64,
    /// (alpha, 正样本 MRR, 负样本 MRR, 差值)
    pub grid: Vec<(f64, f64, f64, f64)>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct LoggedHit {
    chunk_id: i64,
    path: String,
    heading: String,
    cosine: f64,
    lexical: f64,
}

#[derive(Debug, Clone)]
struct Label {
    chunk_id: Option<i64>,
    path: String,
    heading: String,
    positive: bool,
}

/// 用反馈重放两段式打分，网格搜索 α：
/// 正反馈希望目标排得高，负反馈希望排得低。
pub async fn tune(client: &Client, log_dir: &Path, grid_max: f64, step: f64) -> Result<TuneReport> {
    let mut warnings = Vec::new();
    let turns = load_turn_logs(log_dir, &mut warnings)?;
    let labels = load_labels(client).await?;

    let mut grouped: HashMap<(String, i32), Vec<Label>> = HashMap::new();
    for (session_id, turn, label) in labels {
        grouped.entry((session_id, turn)).or_default().push(label);
    }
    let mut positive_events = 0;
    let mut negative_events = 0;
    for labels in grouped.values() {
        for l in labels {
            if l.positive {
                positive_events += 1;
            } else {
                negative_events += 1;
            }
        }
    }
    if positive_events + negative_events < 5 {
        warnings.push("反馈样本少于 5 条，结果只能当流程验证".into());
    }

    let mut best = (0.05, f64::MIN);
    let mut grid = Vec::new();
    let step = step.max(0.01);
    let mut alpha = 0.0f64;
    while alpha <= grid_max + 1e-9 {
        let (pos, neg) = evaluate(&turns, &grouped, alpha);
        let score = pos - neg;
        grid.push((alpha, pos, neg, score));
        if score > best.1 {
            best = (alpha, score);
        }
        alpha += step;
    }

    if positive_events > 0 && negative_events > 0 && best.1.abs() < 1e-9 {
        warnings.push("正负样本在所有 α 下名次一致，当前反馈无法区分权重".into());
        // 没有区分度时不要给出 α=0 这种极端值；保留默认 0.05。
        if let Some((a, _, _, s)) = grid
            .iter()
            .min_by(|x, y| (x.0 - 0.05).abs().partial_cmp(&(y.0 - 0.05).abs()).unwrap())
        {
            best = (*a, *s);
        }
    }

    Ok(TuneReport {
        turns_with_feedback: grouped.len(),
        positive_events,
        negative_events,
        best_alpha: best.0,
        best_score: best.1,
        grid,
        warnings,
    })
}

fn evaluate(
    turns: &HashMap<(String, i32), Vec<LoggedHit>>,
    grouped: &HashMap<(String, i32), Vec<Label>>,
    alpha: f64,
) -> (f64, f64) {
    let mut pos_sum = 0.0;
    let mut pos_n = 0usize;
    let mut neg_sum = 0.0;
    let mut neg_n = 0usize;

    for ((session_id, turn), labels) in grouped {
        let key = (session_id.clone(), *turn);
        let Some(hits) = turns.get(&key) else {
            continue;
        };
        let max_lex = hits
            .iter()
            .map(|h| h.lexical)
            .fold(0.0f64, f64::max)
            .max(1e-6);
        let mut order: Vec<usize> = (0..hits.len()).collect();
        order.sort_by(|&a, &b| {
            let sa = hits[a].cosine + alpha * (hits[a].lexical / max_lex);
            let sb = hits[b].cosine + alpha * (hits[b].lexical / max_lex);
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        for label in labels {
            let Some(pos) = order.iter().position(|&i| {
                if let Some(cid) = label.chunk_id {
                    return hits[i].chunk_id == cid;
                }
                hits[i].path == label.path
                    && (label.heading.is_empty()
                        || hits[i].heading == label.heading
                        || label.heading.ends_with(&hits[i].heading))
            }) else {
                continue;
            };
            let rr = 1.0 / (pos as f64 + 1.0);
            if label.positive {
                pos_sum += rr;
                pos_n += 1;
            } else {
                neg_sum += rr;
                neg_n += 1;
            }
        }
    }

    (
        if pos_n > 0 {
            pos_sum / pos_n as f64
        } else {
            0.0
        },
        if neg_n > 0 {
            neg_sum / neg_n as f64
        } else {
            0.0
        },
    )
}

fn load_turn_logs(
    log_dir: &Path,
    warnings: &mut Vec<String>,
) -> Result<HashMap<(String, i32), Vec<LoggedHit>>> {
    let mut out = HashMap::new();
    let entries = fs::read_dir(log_dir)
        .with_context(|| format!("读取日志目录失败: {}", log_dir.display()))?;
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("jsonl") {
            continue;
        }
        let text = fs::read_to_string(&path).unwrap_or_default();
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if v.get("kind").and_then(|x| x.as_str()) != Some("turn") {
                continue;
            }
            let Some(session_id) = v.get("session_id").and_then(|x| x.as_str()) else {
                continue;
            };
            let Some(turn) = v.get("turn").and_then(|x| x.as_u64()) else {
                continue;
            };
            let mut hits = Vec::new();
            if let Some(arr) = v.pointer("/retrieval/hits").and_then(|x| x.as_array()) {
                for h in arr {
                    let Some(path) = h.get("path").and_then(|x| x.as_str()) else {
                        continue;
                    };
                    let chunk_id = h.get("chunk_id").and_then(|x| x.as_i64()).unwrap_or(0);
                    let heading = h
                        .get("heading")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    let cosine = h
                        .get("vector_score")
                        .and_then(|x| x.as_f64())
                        .unwrap_or(0.0);
                    let lexical = h
                        .get("lexical_score")
                        .and_then(|x| x.as_f64())
                        .unwrap_or(0.0);
                    hits.push(LoggedHit {
                        chunk_id,
                        path: path.to_string(),
                        heading,
                        cosine,
                        lexical,
                    });
                }
            }
            if hits.is_empty() {
                warnings.push(format!("{session_id}/turn {turn} 没有可用的分数字段"));
                continue;
            }
            out.insert((session_id.to_string(), turn as i32), hits);
        }
    }
    Ok(out)
}

async fn load_labels(client: &Client) -> Result<Vec<(String, i32, Label)>> {
    let rows = client
        .query(
            "select session_id, turn, kind, target, detail::text
             from feedback
             where kind in ('adopt', 'reject', 'cite') and session_id is not null",
            &[],
        )
        .await
        .context("读取反馈失败")?;
    let mut labels = Vec::new();
    for r in rows {
        let session_id: String = r.get(0);
        let turn: i32 = r.get(1);
        let kind: String = r.get(2);
        let target: Option<String> = r.get(3);
        let detail_text: String = r.get(4);
        let detail: Value = serde_json::from_str(&detail_text).unwrap_or(Value::Null);
        let sources = detail
            .get("sources")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        match kind.as_str() {
            "cite" => {
                let Some(n) = target.and_then(|t| t.parse::<usize>().ok()) else {
                    continue;
                };
                if let Some(s) = sources
                    .iter()
                    .find(|s| s.get("n").and_then(|x| x.as_u64()) == Some(n as u64))
                {
                    labels.push((session_id.clone(), turn, label_from(s, true)));
                }
            }
            "adopt" => {
                for s in &sources {
                    labels.push((session_id.clone(), turn, label_from(s, true)));
                }
            }
            "reject" => {
                for s in &sources {
                    labels.push((session_id.clone(), turn, label_from(s, false)));
                }
            }
            _ => {}
        }
    }
    Ok(labels)
}

fn label_from(source: &Value, positive: bool) -> Label {
    Label {
        chunk_id: source.get("chunk_id").and_then(|x| x.as_i64()),
        path: source
            .get("path")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        heading: source
            .get("heading")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        positive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(chunk_id: i64, path: &str, cosine: f64, lexical: f64) -> LoggedHit {
        LoggedHit {
            chunk_id,
            path: path.to_string(),
            heading: String::new(),
            cosine,
            lexical,
        }
    }

    #[test]
    fn alpha_can_change_rank() {
        let mut turns = HashMap::new();
        turns.insert(
            ("s".to_string(), 1),
            vec![hit(1, "a", 0.9, 0.0), hit(2, "b", 0.8, 10.0)],
        );
        let mut grouped = HashMap::new();
        grouped.insert(
            ("s".to_string(), 1),
            vec![
                Label {
                    chunk_id: Some(2),
                    path: "b".into(),
                    heading: String::new(),
                    positive: true,
                },
                Label {
                    chunk_id: Some(1),
                    path: "a".into(),
                    heading: String::new(),
                    positive: false,
                },
            ],
        );
        let (p0, n0) = evaluate(&turns, &grouped, 0.0);
        let (p1, n1) = evaluate(&turns, &grouped, 0.5);
        assert!(p1 > p0);
        assert!(n1 < n0);
    }
}
