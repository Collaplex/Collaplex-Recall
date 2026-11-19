use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};

#[derive(Debug, Clone)]
pub struct Entry {
    pub index: usize,
    pub title: String,
    pub body: String,
}

fn read(path: &Path) -> Result<String> {
    fs::read_to_string(path).with_context(|| format!("读取收件箱失败: {}", path.display()))
}

fn write_atomic(path: &Path, content: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).ok();
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, content).with_context(|| format!("写收件箱临时文件失败: {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("替换收件箱失败: {}", path.display()))?;
    Ok(())
}

pub fn parse(content: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut title: Option<String> = None;
    let mut body = String::new();
    for line in content.lines() {
        if let Some(t) = line.strip_prefix("## ") {
            if let Some(prev) = title.take() {
                out.push(Entry {
                    index: out.len() + 1,
                    title: prev,
                    body: body.trim().to_string(),
                });
                body.clear();
            }
            title = Some(t.trim().to_string());
        } else if title.is_some() {
            body.push_str(line);
            body.push('\n');
        }
    }
    if let Some(prev) = title {
        out.push(Entry {
            index: out.len() + 1,
            title: prev,
            body: body.trim().to_string(),
        });
    }
    out
}

pub fn list(path: &Path) -> Result<Vec<Entry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    Ok(parse(&read(path)?))
}

/// 把第 index 条提升到目标笔记（追加），并从收件箱移除。
pub fn promote(path: &Path, index: usize, target: &Path) -> Result<()> {
    let entries = list(path)?;
    let Some(entry) = entries.iter().find(|e| e.index == index) else {
        bail!("收件箱没有第 {index} 条");
    };
    if !target.exists() {
        bail!("目标笔记不存在: {}", target.display());
    }
    let mut t = fs::read_to_string(target)
        .with_context(|| format!("读取目标笔记失败: {}", target.display()))?;
    if !t.ends_with('\n') {
        t.push('\n');
    }
    t.push_str(&format!("\n{}\n", entry.body));
    fs::write(target, t).with_context(|| format!("写入目标笔记失败: {}", target.display()))?;

    let remaining: Vec<Entry> = entries.into_iter().filter(|e| e.index != index).collect();
    write_atomic(path, &rebuild(&remaining))?;
    Ok(())
}

pub fn drop_entry(path: &Path, index: usize) -> Result<bool> {
    let entries = list(path)?;
    if !entries.iter().any(|e| e.index == index) {
        return Ok(false);
    }
    let remaining: Vec<Entry> = entries.into_iter().filter(|e| e.index != index).collect();
    write_atomic(path, &rebuild(&remaining))?;
    Ok(true)
}

fn rebuild(entries: &[Entry]) -> String {
    let mut s = String::new();
    for e in entries {
        s.push_str(&format!("## {}\n\n{}\n\n", e.title, e.body));
    }
    s
}
