use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Clone)]
pub struct RerankConfig {
    pub enabled: bool,
    pub url: String,
    pub model: String,
    pub timeout_ms: u64,
}

impl RerankConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: std::env::var("NOTE_RERANK")
                .map(|v| !matches!(v.to_ascii_lowercase().as_str(), "off" | "0" | "false"))
                .unwrap_or(false),
            url: std::env::var("NOTE_RERANK_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8083/rerank".into()),
            model: std::env::var("NOTE_RERANK_MODEL").unwrap_or_else(|_| "bge".into()),
            timeout_ms: std::env::var("NOTE_RERANK_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3000),
        }
    }
}

#[derive(Debug, Deserialize)]
struct RerankResponse {
    results: Vec<RerankItem>,
}

#[derive(Debug, Deserialize)]
struct RerankItem {
    index: usize,
    relevance_score: f64,
}

/// 交叉编码器打分：返回 (原下标, 分数)，按分数降序。
pub async fn rerank(
    cfg: &RerankConfig,
    query: &str,
    documents: &[String],
) -> Result<(Vec<(usize, f64)>, u128)> {
    if documents.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let started = Instant::now();
    let http = reqwest::Client::builder()
        .timeout(Duration::from_millis(cfg.timeout_ms.max(500)))
        .build()
        .context("构造精排 HTTP 客户端失败")?;
    let body = json!({
        "model": cfg.model,
        "query": query,
        "documents": documents,
    });
    let resp = http
        .post(&cfg.url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("精排请求失败: {}", cfg.url))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("精排服务返回 {status}: {text}");
    }
    let parsed: RerankResponse = resp.json().await.context("解析精排响应失败")?;
    if parsed.results.len() != documents.len() {
        bail!(
            "精排返回 {} 条，期望 {} 条",
            parsed.results.len(),
            documents.len()
        );
    }
    let mut scores: Vec<(usize, f64)> = parsed
        .results
        .into_iter()
        .map(|r| (r.index, r.relevance_score))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok((scores, started.elapsed().as_millis()))
}
