use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::chat::ChatConfig;
use crate::search::Hit;

#[derive(Debug, Clone)]
pub struct SelectConfig {
    pub enabled: bool,
    pub max_pick: usize,
    pub timeout_ms: u64,
}

impl SelectConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: std::env::var("NOTE_SELECT")
                .map(|v| !matches!(v.to_ascii_lowercase().as_str(), "off" | "0" | "false"))
                .unwrap_or(true),
            max_pick: std::env::var("NOTE_SELECT_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            timeout_ms: std::env::var("NOTE_SELECT_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3000),
        }
    }
}

pub struct SelectResult {
    /// 候选里的 0 基下标，按照模型给出的相关性顺序。
    pub picks: Vec<usize>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub elapsed_ms: u128,
}

const SYSTEM: &str = "你是检索选材器。选出所有对回答有帮助的片段，按相关性排序；通常选 2–5 条，\
宁多勿漏，但无关的不要选。不要改写、不要总结、不要补内容。\
只输出一个 JSON 对象：{\"pick\":[1,3,5]}，编号从 1 开始，最多按提示条数。";

/// 第 2 遍的模型部分：只做选择，输出编号；预算和截断由程序负责。
pub async fn select(
    cfg: &ChatConfig,
    sc: &SelectConfig,
    question: &str,
    candidates: &[&Hit],
) -> Result<SelectResult> {
    if candidates.is_empty() {
        bail!("没有候选片段");
    }
    let http = reqwest::Client::builder()
        .timeout(Duration::from_millis(sc.timeout_ms.max(500)))
        .build()
        .context("构造选材 HTTP 客户端失败")?;

    let mut list = String::new();
    for (i, h) in candidates.iter().enumerate() {
        let head = if h.heading.is_empty() {
            h.path.clone()
        } else {
            format!("{} › {}", h.path, h.heading)
        };
        let snip: String = h
            .text
            .chars()
            .take(60)
            .map(|c| if c == '\n' { ' ' } else { c })
            .collect();
        list.push_str(&format!("[{}] {head}｜{snip}\n", i + 1));
    }

    let user = format!(
        "最多选 {} 条。\n用户问题：{question}\n候选：\n{list}",
        sc.max_pick
    );
    let body = json!({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}
        ],
        "temperature": 0.0,
        "max_tokens": 80,
        "stream": false,
    });
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let mut req = http.post(&url).json(&body);
    if let Some(key) = &cfg.api_key {
        req = req.bearer_auth(key);
    }

    let started = Instant::now();
    let resp = req.send().await.context("选材请求失败")?;
    if !resp.status().is_success() {
        bail!("选材服务返回 {}", resp.status());
    }
    let v: Value = resp.json().await.context("解析选材响应失败")?;
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .context("选材响应没有 content")?;
    let picks = parse_picks(content, candidates.len(), sc.max_pick)?;
    if picks.is_empty() {
        bail!("选材结果为空");
    }

    Ok(SelectResult {
        picks,
        prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

/// 接受 {"pick":[1,3]}，也容忍模型写成 {"ids":[...]}；越界和重复直接丢弃。
fn parse_picks(content: &str, n_candidates: usize, max_pick: usize) -> Result<Vec<usize>> {
    let start = content.find('{').context("选材结果不是 JSON")?;
    let end = content.rfind('}').context("选材结果不完整")?;
    let v: Value = serde_json::from_str(&content[start..=end]).context("选材 JSON 解析失败")?;
    let arr = v
        .get("pick")
        .or_else(|| v.get("ids"))
        .and_then(|x| x.as_array())
        .context("选材 JSON 缺少 pick 数组")?;

    let mut out: Vec<usize> = Vec::new();
    for x in arr {
        let Some(n) = x.as_u64() else { continue };
        let idx = n as usize;
        if idx == 0 || idx > n_candidates {
            continue;
        }
        let idx = idx - 1;
        if !out.contains(&idx) {
            out.push(idx);
        }
        if out.len() >= max_pick {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_dedupes_out_of_range() {
        let picks = parse_picks(r#"{"pick":[3,1,3,9,0]}"#, 5, 8).unwrap();
        assert_eq!(picks, vec![2, 0]);
    }

    #[test]
    fn accepts_ids_alias_and_caps() {
        let picks = parse_picks(r#"{"ids":[1,2,3,4]}"#, 4, 2).unwrap();
        assert_eq!(picks, vec![0, 1]);
    }
}
