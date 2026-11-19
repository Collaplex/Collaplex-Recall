use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Local;
use serde_json::Value;

/// 每轮二层的落盘日志：设计里唯一的复现能力，第一天就要有。
/// v0 只记检索（查询、候选、名次、最终命中），够回放“为什么这次召回了这些”。
#[derive(Clone)]
pub struct SessionLog {
    path: PathBuf,
}

impl SessionLog {
    pub fn new(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir).with_context(|| format!("创建日志目录失败: {}", dir.display()))?;
        let name = format!("session-{}.jsonl", Local::now().format("%Y%m%d-%H%M%S"));
        Ok(Self {
            path: dir.join(name),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&self, value: &Value) -> Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("打开日志失败: {}", self.path.display()))?;
        writeln!(f, "{}", serde_json::to_string(value)?).context("写日志失败")?;
        Ok(())
    }
}
