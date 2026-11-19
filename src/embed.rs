use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// 向量来源可插拔：v0 只需要「能出固定维度向量」这一件事。
/// 换模型 = 换 model_id，旧向量留在库里不冲突（检索按 model 过滤）。
#[async_trait]
pub trait Embedder: Send + Sync {
    fn model_id(&self) -> String;
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// 查询侧单独一个入口：有些嵌入模型要求 query 带 instruction 前缀，
    /// 文档侧不带（Qwen3-Embedding、bge 系列都是这样）。
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let mut v = self.embed(&[query.to_string()]).await?;
        v.pop().context("嵌入结果为空")
    }
}

pub fn vector_literal(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 10 + 2);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        // 避免 "NaN" / "inf" 这类 Postgres 不接受的写法
        if x.is_finite() {
            s.push_str(&format!("{x:.6}"));
        } else {
            s.push('0');
        }
    }
    s.push(']');
    s
}

// ---------- 离线兜底：哈希向量 ----------

pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(16) }
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
    fn model_id(&self) -> String {
        format!("hash-v1-{}", self.dim)
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| hash_embed(t, self.dim)).collect())
    }
}

fn hash_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    for token in crate::token::terms(text) {
        let h = Sha256::digest(token.as_bytes());
        let idx = u64::from_le_bytes(h[..8].try_into().unwrap()) as usize % dim;
        let sign = if h[8] & 1 == 0 { 1.0 } else { -1.0 };
        v[idx] += sign;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    } else {
        v[0] = 1.0;
    }
    v
}

// ---------- HTTP：任何 OpenAI 兼容 /v1/embeddings ----------

pub struct HttpEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
    expect_dim: Option<usize>,
    api_key: Option<String>,
    query_prefix: Option<String>,
    doc_prefix: Option<String>,
}

impl HttpEmbedder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_url: String,
        model: String,
        expect_dim: Option<usize>,
        api_key: Option<String>,
        query_prefix: Option<String>,
        doc_prefix: Option<String>,
        timeout_secs: u64,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .context("构造 HTTP 客户端失败")?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
            expect_dim,
            api_key,
            query_prefix,
            doc_prefix,
        })
    }

    /// 前缀算进 model_id：改了前缀就是另一套向量，ingest 会自动补。
    fn prefix_model_id(&self) -> String {
        if self.query_prefix.is_none() && self.doc_prefix.is_none() {
            return self.model.clone();
        }
        let mut h = Sha256::new();
        h.update(self.model.as_bytes());
        if let Some(p) = &self.query_prefix {
            h.update(p.as_bytes());
        }
        h.update([0u8]);
        if let Some(p) = &self.doc_prefix {
            h.update(p.as_bytes());
        }
        let hex = format!("{:x}", h.finalize());
        format!("{}@{}", self.model, &hex[..8])
    }

    async fn embed_raw(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        const BATCH: usize = 32;
        let mut out = Vec::with_capacity(texts.len());

        for batch in texts.chunks(BATCH) {
            let body = serde_json::json!({ "model": self.model, "input": batch });
            let url = format!("{}/embeddings", self.base_url);
            let mut req = self.client.post(&url).json(&body);
            if let Some(key) = &self.api_key {
                req = req.bearer_auth(key);
            }
            let resp = req
                .send()
                .await
                .with_context(|| format!("请求嵌入服务失败: {url}"))?;
            let status = resp.status();
            if !status.is_success() {
                let text = resp.text().await.unwrap_or_default();
                bail!("嵌入服务返回 {status}: {text}");
            }
            let mut parsed: EmbedResponse = resp.json().await.context("解析嵌入响应失败")?;
            parsed.data.sort_by_key(|d| d.index.unwrap_or(usize::MAX));
            if parsed.data.len() != batch.len() {
                bail!(
                    "嵌入服务返回 {} 条，期望 {} 条",
                    parsed.data.len(),
                    batch.len()
                );
            }
            for d in parsed.data {
                if let Some(dim) = self.expect_dim
                    && d.embedding.len() != dim
                {
                    bail!(
                        "嵌入维度不符：期望 {dim}，实际 {}（检查 NOTE_EMBED_DIM / 模型）",
                        d.embedding.len()
                    );
                }
                out.push(d.embedding);
            }
        }
        Ok(out)
    }
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
    #[serde(default)]
    index: Option<usize>,
}

#[async_trait]
impl Embedder for HttpEmbedder {
    fn model_id(&self) -> String {
        self.prefix_model_id()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match &self.doc_prefix {
            Some(p) => {
                let prepared: Vec<String> = texts.iter().map(|t| format!("{p}{t}")).collect();
                self.embed_raw(&prepared).await
            }
            None => self.embed_raw(texts).await,
        }
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let prepared = match &self.query_prefix {
            Some(p) => format!("{p}{query}"),
            None => query.to_string(),
        };
        let mut v = self.embed_raw(&[prepared]).await?;
        v.pop().context("嵌入结果为空")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_vectors_are_deterministic_and_normalized() {
        let e = HashEmbedder::new(128);
        let a = rt_block(e.embed(&["校园网认证".to_string()])).unwrap();
        let b = rt_block(e.embed(&["校园网认证".to_string()])).unwrap();
        assert_eq!(a[0], b[0]);
        let norm: f32 = a[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn similar_texts_score_higher_than_unrelated() {
        let e = HashEmbedder::new(512);
        let vs = rt_block(e.embed(&[
            "校园网认证掉线".to_string(),
            "校园网认证突然中断".to_string(),
            "显卡显存不够".to_string(),
        ]))
        .unwrap();
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f32>();
        assert!(dot(&vs[0], &vs[1]) > dot(&vs[0], &vs[2]));
    }

    fn rt_block<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }
}
