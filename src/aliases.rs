use anyhow::{Context, Result, bail};
use serde::Serialize;
use tokio_postgres::Client;

/// 别名归一化：去首尾空白、折叠空白、ASCII 小写。
pub fn normalize(s: &str) -> String {
    let mut out = String::new();
    let mut last_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.extend(c.to_lowercase());
            last_space = false;
        }
    }
    out.trim().to_string()
}

#[derive(Debug, PartialEq, Eq)]
pub enum AddOutcome {
    Created,
    Updated,
    Unchanged,
}

#[derive(Debug, Clone, Serialize)]
pub struct AliasRow {
    pub alias: String,
    pub entity: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AliasHit {
    pub alias: String,
    pub entity: String,
}

/// 同一个别名只能指向一个实体；不同实体默认拒绝，--force 才覆盖。
pub async fn add(client: &Client, alias: &str, entity: &str, force: bool) -> Result<AddOutcome> {
    let alias = alias.trim();
    let entity = entity.trim();
    let norm = normalize(alias);
    if norm.is_empty() {
        bail!("别名不能为空");
    }
    if entity.is_empty() {
        bail!("实体不能为空");
    }

    let existing = client
        .query_opt("select entity from aliases where normalized = $1", &[&norm])
        .await
        .context("查询别名失败")?;

    match existing {
        Some(row) => {
            let old: String = row.get(0);
            if old == entity {
                return Ok(AddOutcome::Unchanged);
            }
            if !force {
                bail!("冲突：别名「{alias}」已指向「{old}」（要覆盖加 --force）");
            }
            client
                .execute(
                    "update aliases set alias = $1, entity = $2, updated_at = now()
                     where normalized = $3",
                    &[&alias, &entity, &norm],
                )
                .await
                .context("更新别名失败")?;
            Ok(AddOutcome::Updated)
        }
        None => {
            client
                .execute(
                    "insert into aliases (alias, normalized, entity)
                     values ($1, $2, $3)
                     on conflict (normalized) do update
                       set alias = excluded.alias,
                           entity = excluded.entity,
                           updated_at = now()",
                    &[&alias, &norm, &entity],
                )
                .await
                .context("写入别名失败")?;
            Ok(AddOutcome::Created)
        }
    }
}

pub async fn list(client: &Client, filter: Option<&str>) -> Result<Vec<AliasRow>> {
    let rows = match filter.map(normalize) {
        Some(f) if !f.is_empty() => {
            let like = format!("%{f}%");
            client
                .query(
                    "select alias, entity, source from aliases
                     where normalized like $1 or entity ilike $1
                     order by alias",
                    &[&like],
                )
                .await
                .context("查询别名失败")?
        }
        _ => client
            .query(
                "select alias, entity, source from aliases order by alias",
                &[],
            )
            .await
            .context("查询别名失败")?,
    };
    Ok(rows
        .iter()
        .map(|r| AliasRow {
            alias: r.get(0),
            entity: r.get(1),
            source: r.get(2),
        })
        .collect())
}

pub async fn remove(client: &Client, alias: &str) -> Result<bool> {
    let norm = normalize(alias);
    if norm.is_empty() {
        bail!("别名不能为空");
    }
    let n = client
        .execute("delete from aliases where normalized = $1", &[&norm])
        .await
        .context("删除别名失败")?;
    Ok(n > 0)
}

/// 查询里出现别名就追加实体，只做确定性展开，不改写原问题。
pub async fn expand(client: &Client, query: &str) -> Result<(String, Vec<AliasHit>)> {
    let q = normalize(query);
    let rows = client
        .query("select alias, normalized, entity from aliases", &[])
        .await
        .context("读取别名表失败")?;

    let mut hits: Vec<AliasHit> = Vec::new();
    for r in rows {
        let norm: String = r.get(1);
        if !norm.is_empty() && q.contains(&norm) {
            hits.push(AliasHit {
                alias: r.get(0),
                entity: r.get(2),
            });
        }
    }
    hits.sort_by(|a, b| a.alias.cmp(&b.alias));
    hits.dedup_by(|a, b| a.alias == b.alias && a.entity == b.entity);

    if hits.is_empty() {
        return Ok((query.to_string(), hits));
    }
    let mut expanded = query.to_string();
    for h in &hits {
        expanded.push(' ');
        expanded.push_str(&h.entity);
    }
    Ok((expanded, hits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_spaces_and_case() {
        assert_eq!(normalize("  RTX  5060 "), "rtx 5060");
        assert_eq!(normalize("校园网"), "校园网");
    }
}
