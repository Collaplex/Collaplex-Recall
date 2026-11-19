use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::chat::ChatConfig;
use crate::session::SessionPointer;

#[derive(Debug, Clone)]
pub struct RewriteConfig {
    pub enabled: bool,
    pub max_tokens: u32,
    pub timeout_ms: u64,
}

impl RewriteConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: std::env::var("NOTE_REWRITE")
                .map(|v| !matches!(v.to_ascii_lowercase().as_str(), "off" | "0" | "false"))
                .unwrap_or(true),
            max_tokens: std::env::var("NOTE_REWRITE_MAX_TOKENS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(120),
            timeout_ms: std::env::var("NOTE_REWRITE_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3000),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Rewritten {
    pub query: String,
    #[serde(default)]
    pub intent: String,
    #[serde(default)]
    pub entities: Vec<String>,
    /// 用户私有简称/外号，且无法从上轮命中或常识确定。用于触发反问。
    #[serde(default)]
    pub unknown: Vec<String>,
}

pub struct RewriteResult {
    pub data: Rewritten,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub elapsed_ms: u128,
}

const SYSTEM: &str = "你是检索查询改写器。根据用户问题和上轮命中，只输出一个 JSON 对象，\
不要解释、不要 Markdown。字段：query（补全指代后的检索短查询，保留原问题关键词）、\
intent（不超过 10 个字）、entities（字符串数组）、unknown（问题里像用户自己起的名字/\
代号/简称、不是通用技术词或常见名词的词，最多 1 个；即使你能猜到它指什么，也要列出来\
让用户确认；没有就给空数组）。";

/// 独立的非流式小请求：结构化输出、有超时，失败由调用方降级。
pub async fn rewrite(
    cfg: &ChatConfig,
    rc: &RewriteConfig,
    question: &str,
    pointer: &SessionPointer,
) -> Result<RewriteResult> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_millis(rc.timeout_ms.max(500)))
        .build()
        .context("构造判题 HTTP 客户端失败")?;
    let last = pointer
        .last_hits
        .first()
        .map(|h| {
            if h.heading.is_empty() {
                h.path.clone()
            } else {
                format!("{} › {}", h.path, h.heading)
            }
        })
        .unwrap_or_else(|| "（无）".to_string());
    let body = json!({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": format!("上轮命中：{last}\n用户问题：{question}")}
        ],
        "temperature": 0.0,
        "max_tokens": rc.max_tokens,
        "stream": false,
    });
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let mut req = http.post(&url).json(&body);
    if let Some(key) = &cfg.api_key {
        req = req.bearer_auth(key);
    }

    let started = Instant::now();
    let resp = req.send().await.context("判题请求失败")?;
    if !resp.status().is_success() {
        bail!("判题服务返回 {}", resp.status());
    }
    let v: Value = resp.json().await.context("解析判题响应失败")?;
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .context("判题响应没有 content")?;
    let data = parse_rewritten(content)?;
    if data.query.trim().is_empty() {
        bail!("判题结果 query 为空");
    }

    Ok(RewriteResult {
        data,
        prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn parse_rewritten(content: &str) -> Result<Rewritten> {
    let start = content.find('{').context("判题结果不是 JSON")?;
    let end = content.rfind('}').context("判题结果不完整")?;
    let json_str = &content[start..=end];
    let v: Rewritten = serde_json::from_str(json_str).context("判题 JSON 字段不符")?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fenced_json() {
        let s = "```json\n{\"query\":\"校园网 掉线 周期\",\"intent\":\"排障\",\"entities\":[\"校园网\"],\"unknown\":[\"小破机\"]}\n```";
        let r = parse_rewritten(s).unwrap();
        assert_eq!(r.query, "校园网 掉线 周期");
        assert_eq!(r.intent, "排障");
        assert_eq!(r.entities, vec!["校园网"]);
        assert_eq!(r.unknown, vec!["小破机"]);
    }

    #[test]
    fn rejects_non_json() {
        assert!(parse_rewritten("不知道").is_err());
    }
}
