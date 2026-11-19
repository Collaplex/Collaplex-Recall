use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use tokio_postgres::Client;

use crate::chunk::sha256_hex;
use crate::token;

#[derive(Debug, Clone, Serialize)]
pub struct ExperienceRow {
    pub id: i64,
    pub text: String,
    pub status: String,
    pub uses: i32,
}

pub enum AddOutcome {
    Created(i64),
    Existing(i64),
}

/// 候选经验需要连续相关多少轮才自动晋升。
pub fn promote_after() -> i32 {
    std::env::var("NOTE_EXPERIENCE_PROMOTE_USES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

/// 经验只进候选区；复用验证通过才由 promote 或 touch_candidates 晋升。
pub async fn add_candidate(
    client: &Client,
    text: &str,
    session_id: &str,
    turn: i32,
) -> Result<AddOutcome> {
    let text = text.trim();
    if text.is_empty() || text.chars().count() > 200 {
        bail!("经验为空或超过 200 字");
    }
    let hash = sha256_hex(text);
    let existing = client
        .query_opt(
            "select id from experiences where text_hash = $1 and status <> 'rejected'",
            &[&hash],
        )
        .await
        .context("检查重复经验失败")?;
    if let Some(row) = existing {
        return Ok(AddOutcome::Existing(row.get(0)));
    }
    let row = client
        .query_one(
            "insert into experiences (text, text_hash, source_session, source_turn)
             values ($1, $2, $3, $4)
             returning id",
            &[&text, &hash, &session_id, &turn],
        )
        .await
        .context("写入经验候选失败")?;
    Ok(AddOutcome::Created(row.get(0)))
}

pub async fn list(client: &Client) -> Result<Vec<ExperienceRow>> {
    let rows = client
        .query(
            "select id, text, status, uses
             from experiences
             where status <> 'rejected'
             order by (status = 'active') desc, id",
            &[],
        )
        .await
        .context("读取经验失败")?;
    Ok(rows
        .iter()
        .map(|r| ExperienceRow {
            id: r.get(0),
            text: r.get(1),
            status: r.get(2),
            uses: r.get(3),
        })
        .collect())
}

pub async fn promote(client: &Client, id: i64) -> Result<bool> {
    let n = client
        .execute(
            "update experiences
             set status = 'active', promoted_at = now()
             where id = $1 and status = 'candidate'",
            &[&id],
        )
        .await
        .context("晋升经验失败")?;
    Ok(n > 0)
}

pub async fn reject(client: &Client, id: i64) -> Result<bool> {
    let n = client
        .execute(
            "update experiences set status = 'rejected' where id = $1 and status <> 'rejected'",
            &[&id],
        )
        .await
        .context("拒绝经验失败")?;
    Ok(n > 0)
}

/// 进系统提示的恒定 skill 尾部；只取 active，数量有上限。
pub async fn active(client: &Client, limit: usize) -> Result<Vec<String>> {
    let rows = client
        .query(
            "select text from experiences
             where status = 'active'
             order by promoted_at nulls last, id
             limit $1",
            &[&(limit as i64)],
        )
        .await
        .context("读取已晋升经验失败")?;
    Ok(rows.iter().map(|r| r.get(0)).collect())
}

/// 复用验证：候选经验与后续问题的词项重合达到阈值就记一次；
/// 达到 promote_after 次自动晋升。返回本次新晋升的文本。
/// 找出与当前问题相关、但还没晋升的经验候选；只匹配，不改计数。
/// 计数由真实反馈事件（采纳/引用为 +1，纠正为 -1）驱动。
pub async fn match_candidates(client: &Client, query: &str) -> Result<Vec<(i64, String)>> {
    let qterms: HashSet<String> = token::terms(query).into_iter().collect();
    if qterms.is_empty() {
        return Ok(Vec::new());
    }
    let rows = client
        .query(
            "select id, text from experiences where status = 'candidate'",
            &[],
        )
        .await
        .context("读取经验候选失败")?;

    let mut matched = Vec::new();
    for r in rows {
        let id: i64 = r.get(0);
        let text: String = r.get(1);
        let terms: HashSet<String> = token::terms(&text).into_iter().collect();
        let inter = terms.intersection(&qterms).count();
        if inter == 0 {
            continue;
        }
        let ratio = inter as f64 / terms.len().min(qterms.len()).max(1) as f64;
        if ratio >= 0.3 {
            matched.push((id, text));
        }
    }
    Ok(matched)
}

/// 正反馈：候选经验复用计数 +1，达到阈值晋升。
pub async fn reward(client: &Client, ids: &[i64], promote_after: i32) -> Result<Vec<String>> {
    if ids.is_empty() || promote_after <= 0 {
        return Ok(Vec::new());
    }
    let rows = client
        .query(
            "select id, text, uses from experiences
             where status = 'candidate' and id = any($1::bigint[])",
            &[&ids],
        )
        .await
        .context("读取经验候选失败")?;

    let mut promoted = Vec::new();
    for r in rows {
        let id: i64 = r.get(0);
        let text: String = r.get(1);
        let uses: i32 = r.get::<_, i32>(2) + 1;
        if uses >= promote_after {
            client
                .execute(
                    "update experiences
                     set uses = $1, status = 'active', promoted_at = now()
                     where id = $2",
                    &[&uses, &id],
                )
                .await
                .context("晋升经验失败")?;
            promoted.push(text);
        } else {
            client
                .execute(
                    "update experiences set uses = $1 where id = $2",
                    &[&uses, &id],
                )
                .await
                .context("更新经验复用次数失败")?;
        }
    }
    Ok(promoted)
}

/// 负反馈：候选经验复用计数 -1（不低于 0）。
pub async fn punish(client: &Client, ids: &[i64]) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    client
        .execute(
            "update experiences
             set uses = greatest(uses - 1, 0)
             where status = 'candidate' and id = any($1::bigint[])",
            &[&ids],
        )
        .await
        .context("更新经验负反馈失败")?;
    Ok(())
}
