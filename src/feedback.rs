use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use tokio_postgres::Client;

#[derive(Debug, Clone, Serialize)]
pub struct FeedbackRow {
    pub id: i64,
    pub session_id: Option<String>,
    pub turn: Option<i32>,
    pub kind: String,
    pub target: Option<String>,
    pub created_at: String,
    pub detail: Value,
}

/// 反馈只追加，不改写；kind: adopt / reject / cite / delete_memory。
pub async fn record(
    client: &Client,
    session_id: Option<&str>,
    turn: Option<i32>,
    kind: &str,
    target: Option<&str>,
    detail: Value,
) -> Result<i64> {
    let text = detail.to_string();
    let row = client
        .query_one(
            "insert into feedback (session_id, turn, kind, target, detail)
             values ($1, $2, $3, $4, ($5::text)::jsonb)
             returning id",
            &[&session_id, &turn, &kind, &target, &text],
        )
        .await
        .context("写入反馈失败")?;
    Ok(row.get(0))
}

pub async fn list(client: &Client, limit: i64) -> Result<Vec<FeedbackRow>> {
    let rows = client
        .query(
            "select id, session_id, turn, kind, target, created_at::text, detail::text
             from feedback
             order by id desc
             limit $1",
            &[&limit],
        )
        .await
        .context("读取反馈失败")?;
    Ok(rows
        .iter()
        .map(|r| {
            let text: String = r.get(6);
            FeedbackRow {
                id: r.get(0),
                session_id: r.get(1),
                turn: r.get(2),
                kind: r.get(3),
                target: r.get(4),
                created_at: r.get(5),
                detail: serde_json::from_str(&text).unwrap_or(Value::Null),
            }
        })
        .collect())
}
