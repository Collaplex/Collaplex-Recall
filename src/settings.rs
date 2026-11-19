use anyhow::{Context, Result};
use tokio_postgres::Client;

pub async fn get(client: &Client, key: &str) -> Result<Option<String>> {
    let row = client
        .query_opt("select value from settings where key = $1", &[&key])
        .await
        .context("读取设置失败")?;
    Ok(row.map(|r| r.get(0)))
}

pub async fn set(client: &Client, key: &str, value: &str) -> Result<()> {
    client
        .execute(
            "insert into settings (key, value, updated_at)
             values ($1, $2, now())
             on conflict (key) do update
               set value = excluded.value, updated_at = now()",
            &[&key, &value],
        )
        .await
        .context("写入设置失败")?;
    Ok(())
}
