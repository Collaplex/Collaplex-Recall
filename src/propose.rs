use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use similar::TextDiff;
use tokio_postgres::Client;

use crate::chat::ChatConfig;
use crate::chunk::sha256_hex;
use crate::settings;

#[derive(Debug, Clone)]
pub struct Proposal {
    pub id: i64,
    pub note_path: String,
    pub diff: String,
}

#[derive(Debug, Deserialize)]
struct Draft {
    new_content: String,
}

const SYSTEM: &str = "你是补丁提案器。根据用户指令修改目标笔记，只输出 JSON：\
{\"new_content\":\"...\"}，给出修改后的完整笔记内容。保持无关部分逐字不变；\
不要解释、不要 Markdown 代码围栏。";

/// 只读取原文件并生成提案，不写笔记。
pub async fn create(
    client: &Client,
    cfg: &ChatConfig,
    note_path: &str,
    instruction: &str,
    session_id: &str,
    turn: i32,
) -> Result<Proposal> {
    let root = notes_root(client).await?;
    let file = root.join(note_path);
    let old = std::fs::read_to_string(&file)
        .with_context(|| format!("读取笔记失败: {}", file.display()))?;
    if old.chars().count() > 12000 {
        bail!("笔记太长（>12000 字），先缩小范围再提案");
    }
    let old_hash = sha256_hex(&old);

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .context("构造提案 HTTP 客户端失败")?;
    let user = format!("用户指令：{instruction}\n\n当前笔记全文：\n{old}");
    let body = json!({
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
        "stream": false,
    });
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let mut req = http.post(&url).json(&body);
    if let Some(key) = &cfg.api_key {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.context("提案请求失败")?;
    if !resp.status().is_success() {
        bail!("提案服务返回 {}", resp.status());
    }
    let v: Value = resp.json().await.context("解析提案响应失败")?;
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .context("提案响应没有 content")?;
    let draft = parse_draft(content)?;
    if draft.new_content.trim() == old.trim() {
        bail!("模型认为无需修改");
    }

    let diff = TextDiff::from_lines(&old, &draft.new_content)
        .unified_diff()
        .context_radius(3)
        .header(note_path, note_path)
        .to_string();

    let row = client
        .query_one(
            "insert into proposals
               (note_path, base_hash, old_content, new_content, diff, session_id, turn)
             values ($1, $2, $3, $4, $5, $6, $7)
             returning id",
            &[
                &note_path,
                &old_hash,
                &old,
                &draft.new_content,
                &diff,
                &session_id,
                &turn,
            ],
        )
        .await
        .context("写入提案失败")?;
    Ok(Proposal {
        id: row.get(0),
        note_path: note_path.to_string(),
        diff,
    })
}

/// 用户确认后写文件；写入前再次校验 base_hash。
pub async fn apply(client: &Client, id: i64) -> Result<PathBuf> {
    let row = client
        .query_opt(
            "select note_path, base_hash, new_content
             from proposals where id = $1 and status = 'pending'",
            &[&id],
        )
        .await
        .context("读取提案失败")?;
    let Some(row) = row else {
        bail!("提案 #{id} 不存在或不是 pending");
    };
    let note_path: String = row.get(0);
    let base_hash: String = row.get(1);
    let new_content: String = row.get(2);

    let root = notes_root(client).await?;
    let file = root.join(&note_path);
    let current = std::fs::read_to_string(&file)
        .with_context(|| format!("读取笔记失败: {}", file.display()))?;
    if sha256_hex(&current) != base_hash {
        bail!("笔记在提案之后被改过，拒绝覆盖；重新生成提案");
    }

    let tmp = file.with_extension("md.propose-tmp");
    std::fs::write(&tmp, new_content.as_bytes())
        .with_context(|| format!("写临时文件失败: {}", tmp.display()))?;
    std::fs::rename(&tmp, &file).with_context(|| format!("替换笔记失败: {}", file.display()))?;

    client
        .execute(
            "update proposals set status = 'applied', applied_at = now() where id = $1",
            &[&id],
        )
        .await
        .context("更新提案状态失败")?;
    Ok(file)
}

pub async fn reject(client: &Client, id: i64) -> Result<bool> {
    let n = client
        .execute(
            "update proposals set status = 'rejected'
             where id = $1 and status = 'pending'",
            &[&id],
        )
        .await
        .context("拒绝提案失败")?;
    Ok(n > 0)
}

pub async fn list(client: &Client, limit: i64) -> Result<Vec<(i64, String, String)>> {
    let rows = client
        .query(
            "select id, note_path, status from proposals
             where status = 'pending' order by id desc limit $1",
            &[&limit],
        )
        .await
        .context("读取提案失败")?;
    Ok(rows
        .iter()
        .map(|r| (r.get(0), r.get(1), r.get(2)))
        .collect())
}

async fn notes_root(client: &Client) -> Result<PathBuf> {
    let root = settings::get(client, "notes_root").await?;
    let Some(root) = root else {
        bail!("没有记录 notes_root；先 ingest 一次笔记目录");
    };
    Ok(PathBuf::from(root))
}

fn parse_draft(content: &str) -> Result<Draft> {
    let start = content.find('{').context("提案结果不是 JSON")?;
    let end = content.rfind('}').context("提案 JSON 不完整")?;
    let v: Draft = serde_json::from_str(&content[start..=end]).context("提案 JSON 字段不符")?;
    if v.new_content.trim().is_empty() {
        bail!("提案内容为空");
    }
    Ok(v)
}

pub fn diff_lines(diff: &str, max: usize) -> Vec<String> {
    diff.lines().take(max).map(|s| s.to_string()).collect()
}

/// 供测试/其他模块判断路径是否在 root 内。
#[allow(dead_code)]
pub fn is_within(root: &Path, file: &Path) -> bool {
    file.starts_with(root)
}
