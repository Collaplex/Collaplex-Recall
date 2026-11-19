use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::chat::ChatConfig;
use crate::extract::Source;

#[derive(Debug, Clone)]
pub struct VerifyConfig {
    pub enabled: bool,
    pub strict: bool,
    pub timeout_ms: u64,
}

impl VerifyConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: std::env::var("NOTE_VERIFY")
                .map(|v| !matches!(v.to_ascii_lowercase().as_str(), "off" | "0" | "false"))
                .unwrap_or(true),
            strict: std::env::var("NOTE_VERIFY_STRICT")
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "on" | "1" | "true"))
                .unwrap_or(false),
            timeout_ms: std::env::var("NOTE_VERIFY_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4000),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct VerifyResult {
    pub cited: Vec<usize>,
    pub invalid: Vec<usize>,
    pub supported: Option<bool>,
    pub reason: String,
}

impl VerifyResult {
    pub fn ok(&self) -> bool {
        !self.cited.is_empty() && self.invalid.is_empty() && self.supported != Some(false)
    }
}

#[derive(Debug, Deserialize)]
struct Judge {
    supported: bool,
    #[serde(default)]
    reason: String,
}

const SYSTEM: &str = "你是答案校验器。只判断答案里的结论是否被给定引用片段支持。\
只输出 JSON：{\"supported\":true,\"reason\":\"不超过30字\"}；不支持就 supported=false。\
不要解释、不要 Markdown。";

/// 先做结构校验（引用编号），再用聊天模型判断引用是否支撑答案。
pub async fn verify(
    cfg: &ChatConfig,
    vc: &VerifyConfig,
    answer: &str,
    sources: &[Source],
) -> VerifyResult {
    let cited = parse_citations(answer);
    let mut invalid: Vec<usize> = cited
        .iter()
        .copied()
        .filter(|n| *n == 0 || *n > sources.len())
        .collect();
    invalid.sort();
    invalid.dedup();

    let mut result = VerifyResult {
        cited,
        invalid,
        supported: None,
        reason: String::new(),
    };
    if !vc.enabled {
        return result;
    }
    if result.cited.is_empty() {
        result.reason = "答案没有引用任何来源".into();
        return result;
    }
    if !result.invalid.is_empty() {
        result.reason = "引用编号超出来源范围".into();
        return result;
    }

    let mut snippets = String::new();
    for n in &result.cited {
        let s = &sources[*n - 1];
        let snip: String = s.text.chars().take(400).collect();
        snippets.push_str(&format!("[{}] {}｜{}\n\n", n, s.head, snip));
    }
    let user = format!(
        "答案：\n{}\n\n引用片段：\n{}",
        answer.chars().take(1500).collect::<String>(),
        snippets
    );
    let http = match reqwest::Client::builder()
        .timeout(Duration::from_millis(vc.timeout_ms.max(500)))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            result.reason = format!("校验客户端失败：{e}");
            return result;
        }
    };
    let body = json!({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}
        ],
        "temperature": 0.0,
        "max_tokens": 120,
        "stream": false,
    });
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let mut req = http.post(&url).json(&body);
    if let Some(key) = &cfg.api_key {
        req = req.bearer_auth(key);
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            result.reason = format!("校验请求失败：{e}");
            return result;
        }
    };
    if !resp.status().is_success() {
        result.reason = format!("校验服务返回 {}", resp.status());
        return result;
    }
    let v: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            result.reason = format!("校验响应解析失败：{e}");
            return result;
        }
    };
    let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    match parse_judge(content) {
        Ok(j) => {
            result.supported = Some(j.supported);
            result.reason = j.reason;
        }
        Err(e) => result.reason = format!("校验结果不符：{e}"),
    }
    result
}

fn parse_judge(content: &str) -> Result<Judge> {
    let start = content.find('{').context("不是 JSON")?;
    let end = content.rfind('}').context("JSON 不完整")?;
    let j: Judge = serde_json::from_str(&content[start..=end]).context("JSON 字段不符")?;
    Ok(j)
}

/// 支持 [1]、[n1]、[2][3] 这些模型常见写法。
pub fn parse_citations(answer: &str) -> Vec<usize> {
    let bytes = answer.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            let mut j = i + 1;
            if j < bytes.len() && (bytes[j] == b'n' || bytes[j] == b'N') {
                j += 1;
            }
            let start = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > start && j < bytes.len() && bytes[j] == b']' {
                if let Ok(n) = std::str::from_utf8(&bytes[start..j])
                    .unwrap_or("")
                    .parse::<usize>()
                {
                    out.push(n);
                }
                i = j;
                continue;
            }
        }
        i += 1;
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_citation_forms() {
        assert_eq!(parse_citations("结论[1]，另见[n2]。"), vec![1, 2]);
        assert_eq!(parse_citations("[3][1][3]"), vec![1, 3]);
        assert!(parse_citations("没有引用").is_empty());
    }
}
