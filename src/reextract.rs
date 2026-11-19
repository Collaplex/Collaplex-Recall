use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_postgres::Client;

use crate::chat::ChatConfig;
use crate::memory;
use crate::token;

#[derive(Debug, Default)]
pub struct ReextractReport {
    pub rewritten: usize,
    pub dropped: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ModelResult {
    keep: bool,
    #[serde(default)]
    text: String,
}

const SYSTEM: &str = "你是记忆重提炼器。旧记忆和当前笔记片段如下。\
若片段仍然支撑旧记忆，输出 JSON {\"keep\":true,\"text\":\"...\"}，text 必须只依据片段、\
可以更新表述但不得编造；若不再支撑，输出 {\"keep\":false}。只输出 JSON，不要解释。";

/// 模型版：对 stale 记忆，用当前笔记片段判断是否仍成立，并产出更新后的记忆。
pub async fn reextract_with_model(
    client: &Client,
    cfg: &ChatConfig,
    limit: usize,
) -> Result<ReextractReport> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .context("构造重提炼 HTTP 客户端失败")?;

    let rows = client
        .query(
            "select m.id, m.text, m.note_id, m.note_path
             from memories m
             where m.status = 'stale' and not m.pinned and m.note_id is not null
             order by m.id
             limit $1",
            &[&(limit as i64)],
        )
        .await
        .context("读取失效记忆失败")?;

    let mut report = ReextractReport::default();
    for r in rows {
        let id: i64 = r.get(0);
        let old_text: String = r.get(1);
        let note_id: i64 = r.get(2);
        let note_path: String = r.get(3);

        let chunks = client
            .query(
                "select id, heading, text from chunks where note_id = $1",
                &[&note_id],
            )
            .await
            .context("读取当前 chunk 失败")?;
        if chunks.is_empty() {
            report.dropped += 1;
            continue;
        }

        let mterms: HashSet<String> = token::terms(&old_text).into_iter().collect();
        let mut ranked: Vec<(f64, i64, String, String)> = chunks
            .iter()
            .map(|c| {
                let cid: i64 = c.get(0);
                let head: String = c.get(1);
                let text: String = c.get(2);
                let cterms: HashSet<String> = token::terms(&text).into_iter().collect();
                let score = if mterms.is_empty() {
                    0.0
                } else {
                    mterms.intersection(&cterms).count() as f64 / mterms.len() as f64
                };
                (score, cid, head, text)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let best = &ranked[0];
        if best.0 < 0.1 {
            report.dropped += 1;
            continue;
        }

        let mut snippets = String::new();
        for (i, (_, cid, head, text)) in ranked.iter().take(2).enumerate() {
            let snip: String = text.chars().take(500).collect();
            snippets.push_str(&format!(
                "[{}] chunk {}｜{}\n{}\n\n",
                i + 1,
                cid,
                head,
                snip
            ));
        }
        let user = format!("旧记忆：{old_text}\n\n当前片段：\n{snippets}");
        let body = json!({
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}
            ],
            "temperature": 0.0,
            "max_tokens": 220,
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
                report.errors.push(format!("#{id} 请求失败：{e:#}"));
                continue;
            }
        };
        if !resp.status().is_success() {
            report
                .errors
                .push(format!("#{id} 服务返回 {}", resp.status()));
            continue;
        }
        let v: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                report.errors.push(format!("#{id} 解析失败：{e:#}"));
                continue;
            }
        };
        let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
        let parsed = match parse_result(content) {
            Ok(x) => x,
            Err(e) => {
                report.errors.push(format!("#{id} 结果不符：{e:#}"));
                continue;
            }
        };

        if !parsed.keep {
            report.dropped += 1;
            continue;
        }
        let text = parsed.text.trim();
        if text.is_empty() || text.chars().count() > 200 {
            report.dropped += 1;
            continue;
        }
        match memory::add(client, text, &note_path, Some(best.1), false).await {
            Ok(_) => {
                client
                    .execute(
                        "update memories
                         set status = 'superseded', updated_at = now()
                         where id = $1",
                        &[&id],
                    )
                    .await
                    .context("标记旧记忆失败")?;
                report.rewritten += 1;
            }
            Err(e) => report.errors.push(format!("#{id} 写入失败：{e:#}")),
        }
    }
    Ok(report)
}

fn parse_result(content: &str) -> Result<ModelResult> {
    let start = content.find('{').context("不是 JSON")?;
    let end = content.rfind('}').context("JSON 不完整")?;
    let v: ModelResult = serde_json::from_str(&content[start..=end]).context("JSON 字段不符")?;
    if v.keep && v.text.trim().is_empty() {
        bail!("keep=true 但 text 为空");
    }
    Ok(v)
}
