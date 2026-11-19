use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_postgres::Client;

use crate::aliases;
use crate::chat::ChatConfig;
use crate::experience;
use crate::memory;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractMode {
    Immediate,
    EveryN(usize),
    /// 空闲多少毫秒后触发
    Idle(u64),
}

#[derive(Debug, Clone)]
pub struct ExtractConfig {
    pub enabled: bool,
    pub mode: ExtractMode,
    pub max_memories: usize,
    pub max_aliases: usize,
    pub timeout_ms: u64,
}

impl ExtractConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: std::env::var("NOTE_EXTRACT")
                .map(|v| !matches!(v.to_ascii_lowercase().as_str(), "off" | "0" | "false"))
                .unwrap_or(true),
            mode: match std::env::var("NOTE_EXTRACT_MODE")
                .unwrap_or_else(|_| "immediate".into())
                .to_ascii_lowercase()
                .as_str()
            {
                "every_n" | "every-n" => ExtractMode::EveryN(
                    std::env::var("NOTE_EXTRACT_EVERY_N")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(3)
                        .max(1),
                ),
                "idle" => ExtractMode::Idle(
                    std::env::var("NOTE_EXTRACT_IDLE_MS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(8000),
                ),
                _ => ExtractMode::Immediate,
            },
            max_memories: std::env::var("NOTE_EXTRACT_MAX_MEMORIES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3),
            max_aliases: std::env::var("NOTE_EXTRACT_MAX_ALIASES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5),
            timeout_ms: std::env::var("NOTE_EXTRACT_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8000),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub chunk_id: i64,
    pub path: String,
    pub head: String,
    pub text: String,
}

#[derive(Debug, Default, Deserialize)]
struct RawExtraction {
    #[serde(default)]
    memories: Vec<RawMemory>,
    #[serde(default)]
    aliases: Vec<RawAlias>,
    #[serde(default)]
    experiences: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawMemory {
    text: String,
    /// 模型可能写 1、"1"、"n1"、"[1]"，统一在 parse_source 里处理。
    #[serde(default)]
    source: Value,
}

fn parse_source(v: &Value) -> usize {
    if let Some(n) = v.as_u64() {
        return n as usize;
    }
    if let Some(s) = v.as_str() {
        let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = digits.parse::<usize>() {
            return n;
        }
    }
    1
}

#[derive(Debug, Deserialize)]
struct RawAlias {
    alias: String,
    entity: String,
}

pub struct ApplyReport {
    pub delta_id: i64,
    pub memories_added: usize,
    pub memories_duplicate: usize,
    pub aliases_added: usize,
    pub aliases_skipped: usize,
    pub experiences_added: usize,
    pub experiences_duplicate: usize,
    pub errors: Vec<String>,
}

const SYSTEM: &str = "你是记忆提炼器。只从问答和引用的笔记片段里提炼可复用的事实与别名，\
不得编造。只输出一个 JSON 对象，不要解释、不要 Markdown。字段：\
memories（数组，每项 {text, source}，source 是片段编号，最多 3 条；同一事实只写一条，\
不要换措辞重复；必须能由片段支撑）、\
aliases（数组，每项 {alias, entity}，最多 5 条；只提炼用户对已有实体的简称/外号，\
如「小破机→RTX 5060」；不要提炼概念定义、类别或解释，宁缺毋滥）、\
experiences（数组，最多 1 条可复用的流程规则/做法；没有就空数组）。";

/// 回答结束后异步跑：只产出候选，不直接写库。
pub async fn extract(
    cfg: &ChatConfig,
    ec: &ExtractConfig,
    question: &str,
    answer: &str,
    sources: &[Source],
) -> Result<Value> {
    if sources.is_empty() {
        bail!("没有来源片段，跳过提炼");
    }
    let http = reqwest::Client::builder()
        .timeout(Duration::from_millis(ec.timeout_ms.max(1000)))
        .build()
        .context("构造提炼 HTTP 客户端失败")?;

    let mut list = String::new();
    for (i, s) in sources.iter().enumerate() {
        let snip: String = s
            .text
            .chars()
            .take(200)
            .map(|c| if c == '\n' { ' ' } else { c })
            .collect();
        list.push_str(&format!("[{}] {}｜{}\n", i + 1, s.head, snip));
    }
    let answer: String = answer.chars().take(1200).collect();
    let user = format!(
        "最多 {} 条记忆、{} 条别名。\n问题：{question}\n回答：{answer}\n片段：\n{list}",
        ec.max_memories, ec.max_aliases
    );
    let body = json!({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}
        ],
        "temperature": 0.0,
        "max_tokens": 500,
        "stream": false,
    });
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let mut req = http.post(&url).json(&body);
    if let Some(key) = &cfg.api_key {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.context("提炼请求失败")?;
    if !resp.status().is_success() {
        bail!("提炼服务返回 {}", resp.status());
    }
    let v: Value = resp.json().await.context("解析提炼响应失败")?;
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .context("提炼响应没有 content")?;
    parse_json(content)
}

fn parse_json(content: &str) -> Result<Value> {
    let start = content.find('{').context("提炼结果不是 JSON")?;
    let end = content.rfind('}').context("提炼结果不完整")?;
    let v: Value = serde_json::from_str(&content[start..=end]).context("提炼 JSON 解析失败")?;
    Ok(v)
}

/// 第一段：把模型产物写成 pending delta（先落盘）。
pub async fn enqueue(
    client: &Client,
    raw: Value,
    sources: &[Source],
    session_id: &str,
    turn: i32,
) -> Result<i64> {
    let payload = json!({
        "session_id": session_id,
        "turn": turn,
        "raw": raw,
        "sources": sources,
    });
    let row = client
        .query_one(
            "insert into memory_deltas (session_id, turn, payload, status)
             values ($1, $2, ($3::text)::jsonb, 'pending')
             returning id",
            &[&session_id, &turn, &payload.to_string()],
        )
        .await
        .context("写入 pending delta 失败")?;
    Ok(row.get(0))
}

/// 第二段：提交点合并。单用户下由提炼任务立刻调用；
/// 崩溃留下的 pending 可由 `recall delta merge` 补跑。
pub async fn merge_one(
    client: &Client,
    id: i64,
    max_memories: usize,
    max_aliases: usize,
) -> Result<ApplyReport> {
    let row = client
        .query_opt(
            "select payload::text from memory_deltas where id = $1 and status = 'pending'",
            &[&id],
        )
        .await
        .context("读取 pending delta 失败")?;
    let Some(row) = row else {
        bail!("delta #{id} 不存在或不是 pending");
    };
    let text: String = row.get(0);
    let v: Value = serde_json::from_str(&text).context("delta payload 解析失败")?;
    let raw = v.get("raw").cloned().unwrap_or(Value::Null);
    let sources: Vec<Source> =
        serde_json::from_value(v.get("sources").cloned().unwrap_or(json!([])))
            .context("delta sources 解析失败")?;
    let session_id = v
        .get("session_id")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let turn = v.get("turn").and_then(|x| x.as_u64()).unwrap_or(0) as i32;

    let mut report = merge_raw(
        client,
        raw,
        &sources,
        &session_id,
        turn,
        max_memories,
        max_aliases,
    )
    .await?;
    report.delta_id = id;

    let result = json!({
        "memories_added": report.memories_added,
        "memories_duplicate": report.memories_duplicate,
        "aliases_added": report.aliases_added,
        "aliases_skipped": report.aliases_skipped,
        "experiences_added": report.experiences_added,
        "experiences_duplicate": report.experiences_duplicate,
        "errors": report.errors,
    });
    let patch = json!({ "result": result }).to_string();
    client
        .execute(
            "update memory_deltas
             set status = 'applied', applied_at = now(),
                 payload = payload || ($2::text)::jsonb
             where id = $1",
            &[&id, &patch],
        )
        .await
        .context("提交 delta 失败")?;
    Ok(report)
}

pub async fn merge_pending(
    client: &Client,
    limit: usize,
    max_memories: usize,
    max_aliases: usize,
) -> Result<Vec<ApplyReport>> {
    let rows = client
        .query(
            "select id from memory_deltas where status = 'pending' order by id limit $1",
            &[&(limit as i64)],
        )
        .await
        .context("读取 pending delta 失败")?;
    let mut out = Vec::new();
    for r in rows {
        let id: i64 = r.get(0);
        out.push(merge_one(client, id, max_memories, max_aliases).await?);
    }
    Ok(out)
}

/// 校验 + 合并：模型产物在这里被程序限量、绑定来源、去重和写库。
async fn merge_raw(
    client: &Client,
    raw: Value,
    sources: &[Source],
    session_id: &str,
    turn: i32,
    max_memories: usize,
    max_aliases: usize,
) -> Result<ApplyReport> {
    let parsed: RawExtraction = serde_json::from_value(raw).context("提炼字段不符")?;

    let mut report = ApplyReport {
        delta_id: 0,
        memories_added: 0,
        memories_duplicate: 0,
        aliases_added: 0,
        aliases_skipped: 0,
        experiences_added: 0,
        experiences_duplicate: 0,
        errors: Vec::new(),
    };

    for m in parsed.memories.iter().take(max_memories.max(1)) {
        let text = m.text.trim();
        if text.is_empty() || text.chars().count() > 200 {
            report.errors.push("记忆为空或超过 200 字".into());
            continue;
        }
        if sources.is_empty() {
            report.errors.push("没有来源片段".into());
            break;
        }
        let idx = parse_source(&m.source).clamp(1, sources.len()) - 1;
        let src = &sources[idx];
        match memory::add(client, text, &src.path, Some(src.chunk_id), false).await {
            Ok(memory::AddOutcome::Created(_)) => report.memories_added += 1,
            Ok(memory::AddOutcome::Existing(_)) => report.memories_duplicate += 1,
            Err(e) => report.errors.push(format!("记忆写入失败：{e:#}")),
        }
    }

    for a in parsed.aliases.iter().take(max_aliases.max(1)) {
        let (alias, entity) = (a.alias.trim(), a.entity.trim());
        let na = aliases::normalize(alias);
        let ne = aliases::normalize(entity);
        // 拒绝空值、相同值，以及“实体只是别名的子串/超串”的冗余映射
        //（例如「等响度补偿」→「等响度」）。
        if na.is_empty() || ne.is_empty() || na == ne || na.contains(&ne) || ne.contains(&na) {
            report.aliases_skipped += 1;
            continue;
        }
        // 实体必须能在来源片段中找到，否则就是模型编的类别/定义。
        let grounded = sources.iter().any(|s| {
            aliases::normalize(&s.text).contains(&ne) || aliases::normalize(&s.head).contains(&ne)
        });
        if !grounded {
            report.aliases_skipped += 1;
            continue;
        }
        match aliases::add(client, alias, entity, false).await {
            Ok(aliases::AddOutcome::Created | aliases::AddOutcome::Updated) => {
                report.aliases_added += 1
            }
            Ok(aliases::AddOutcome::Unchanged) => report.aliases_skipped += 1,
            Err(e) => {
                report.aliases_skipped += 1;
                report.errors.push(format!("别名跳过：{e:#}"));
            }
        }
    }

    for e in parsed.experiences.iter().take(1) {
        let text = e.trim();
        if text.is_empty() || text.chars().count() > 200 {
            continue;
        }
        match experience::add_candidate(client, text, session_id, turn).await {
            Ok(experience::AddOutcome::Created(_)) => report.experiences_added += 1,
            Ok(experience::AddOutcome::Existing(_)) => report.experiences_duplicate += 1,
            Err(e) => report.errors.push(format!("经验写入失败：{e:#}")),
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_loose_source() {
        assert_eq!(parse_source(&json!(2)), 2);
        assert_eq!(parse_source(&json!("n3")), 3);
        assert_eq!(parse_source(&json!("[1]")), 1);
        assert_eq!(parse_source(&json!("x")), 1);
    }
}
