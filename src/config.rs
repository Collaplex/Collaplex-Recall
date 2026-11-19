use std::path::PathBuf;

/// 运行配置全部来自环境变量，默认值面向本机（PG 实例走 unix socket）。
#[derive(Debug, Clone)]
pub struct Config {
    pub database_url: String,
    pub embed: EmbedKind,
    pub log_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub enum EmbedKind {
    /// 离线兜底：按 token（ASCII 词 + 中文单字/双字）哈希到固定维度。
    /// 不追求语义，只保证没有模型时全链路能跑通、可复现。
    Hash { dim: usize },
    /// 任何 OpenAI 兼容的 /v1/embeddings（llama-server、vLLM、Ollama 等）。
    Http {
        base_url: String,
        model: String,
        dim: Option<usize>,
        api_key: Option<String>,
        /// Qwen3-Embedding 之类的查询 instruction，只加在 query 上。
        query_prefix: Option<String>,
        doc_prefix: Option<String>,
    },
}

impl Config {
    pub fn from_env() -> Self {
        let database_url = std::env::var("NOTE_DATABASE_URL").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/var/lib".into());
            let user = std::env::var("USER").unwrap_or_else(|_| "postgres".into());
            format!("host={home}/pgdata/socket user={user} dbname=note_rs")
        });

        let embed = match std::env::var("NOTE_EMBED")
            .unwrap_or_else(|_| "hash".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "http" => EmbedKind::Http {
                base_url: std::env::var("NOTE_EMBED_URL")
                    .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".into()),
                model: std::env::var("NOTE_EMBED_MODEL").unwrap_or_else(|_| "local-embed".into()),
                dim: std::env::var("NOTE_EMBED_DIM")
                    .ok()
                    .and_then(|v| v.parse().ok()),
                api_key: std::env::var("NOTE_EMBED_API_KEY").ok(),
                query_prefix: std::env::var("NOTE_EMBED_QUERY_PREFIX").ok(),
                doc_prefix: std::env::var("NOTE_EMBED_DOC_PREFIX").ok(),
            },
            _ => EmbedKind::Hash {
                dim: std::env::var("NOTE_HASH_DIM")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(512),
            },
        };

        let log_dir = std::env::var("NOTE_LOG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data/logs"));

        Self {
            database_url,
            embed,
            log_dir,
        }
    }
}
