use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub struct ChatConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub ctx: u32,
    pub max_tokens: u32,
    /// 拼进上下文的片段预算（粗估 token）。
    pub context_budget: usize,
}

impl ChatConfig {
    pub fn from_env() -> Self {
        Self {
            base_url: std::env::var("NOTE_CHAT_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8082/v1".into()),
            model: std::env::var("NOTE_CHAT_MODEL").unwrap_or_else(|_| "local-chat".into()),
            api_key: std::env::var("NOTE_CHAT_API_KEY").ok(),
            ctx: std::env::var("NOTE_CHAT_CTX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096),
            max_tokens: std::env::var("NOTE_CHAT_MAX_TOKENS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(512),
            context_budget: std::env::var("NOTE_CONTEXT_BUDGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2000),
        }
    }
}

/// 状态行需要的最小信息：token、耗时、KV、prefix cache。
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatProgress {
    pub tokens: u64,
    pub prompt_tokens: u32,
    pub kv_ctx: u32,
    pub kv_used: u32,
    pub cache_hit: u32,
    pub elapsed_ms: u128,
    pub ttft_ms: Option<u128>,
    /// 本轮检索上下文用量 / 预算（用于总上下文进度条）
    pub context_used: u32,
    pub context_budget: u32,
}

#[derive(Debug)]
pub struct ChatResult {
    pub content: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u64,
    pub ttft_ms: Option<u128>,
    pub total_ms: u128,
}

#[derive(Default)]
struct KvState {
    ctx: AtomicU32,
    prompt: AtomicU32,
    cache: AtomicU32,
    generated: AtomicU64,
}

fn slots_url(base_url: &str) -> String {
    format!(
        "{}/slots",
        base_url
            .trim_end_matches('/')
            .trim_end_matches("/v1")
            .trim_end_matches('/')
    )
}

/// llama-server 的 /slots 是唯一能拿到预填进度和 prefix cache 命中的地方。
async fn poll_slots(http: reqwest::Client, url: String, kv: Arc<KvState>, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::Relaxed) {
        if let Ok(resp) = http
            .get(&url)
            .timeout(Duration::from_millis(500))
            .send()
            .await
            && let Ok(v) = resp.json::<Value>().await
            && let Some(slots) = v.as_array()
        {
            let slot = slots
                .iter()
                .find(|s| s["is_processing"].as_bool().unwrap_or(false))
                .or_else(|| slots.first());
            if let Some(s) = slot {
                let ctx = s["n_ctx"].as_u64().unwrap_or(0) as u32;
                let prompt = s["n_prompt_tokens"].as_u64().unwrap_or(0) as u32;
                let cache = s["n_prompt_tokens_cache"].as_u64().unwrap_or(0) as u32;
                if ctx > 0 {
                    kv.ctx.store(ctx, Ordering::Relaxed);
                }
                kv.prompt.store(prompt, Ordering::Relaxed);
                kv.cache.store(cache, Ordering::Relaxed);
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn snapshot(started: Instant, generated: u64, ttft: Option<u128>, kv: &KvState) -> ChatProgress {
    let prompt = kv.prompt.load(Ordering::Relaxed);
    ChatProgress {
        tokens: generated,
        prompt_tokens: prompt,
        kv_ctx: kv.ctx.load(Ordering::Relaxed),
        kv_used: prompt + generated as u32,
        cache_hit: kv.cache.load(Ordering::Relaxed),
        elapsed_ms: started.elapsed().as_millis(),
        ttft_ms: ttft,
        context_used: 0,
        context_budget: 0,
    }
}

/// OpenAI 兼容流式聊天。内容只累加不显示，由调用方一次性渲染；
/// 这里只负责 token 计数、计时和 KV 状态。
pub async fn stream_chat<F>(
    cfg: &ChatConfig,
    messages: Vec<Value>,
    mut on_progress: F,
) -> Result<ChatResult>
where
    F: FnMut(ChatProgress) + Send,
{
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .context("构造 HTTP 客户端失败")?;
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let body = json!({
        "model": cfg.model,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
        "temperature": 0.3,
        "max_tokens": cfg.max_tokens,
    });

    let mut req = http.post(&url).json(&body);
    if let Some(key) = &cfg.api_key {
        req = req.bearer_auth(key);
    }

    let started = Instant::now();
    let resp = req
        .send()
        .await
        .with_context(|| format!("连接聊天服务失败: {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        bail!("聊天服务返回 {status}: {text}");
    }

    let kv = Arc::new(KvState::default());
    let stop = Arc::new(AtomicBool::new(false));
    let poller = tokio::spawn(poll_slots(
        http.clone(),
        slots_url(&cfg.base_url),
        kv.clone(),
        stop.clone(),
    ));

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut generated: u64 = 0;
    let mut ttft: Option<u128> = None;
    let mut usage_prompt: u32 = 0;
    let mut usage_completion: u64 = 0;
    let mut last_report = Instant::now();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("读取聊天流失败")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].trim_end_matches('\r').to_string();
            buf.drain(..=pos);

            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };

            if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                usage_prompt = u["prompt_tokens"].as_u64().unwrap_or(0) as u32;
                usage_completion = u["completion_tokens"].as_u64().unwrap_or(0);
            }

            let delta = &v["choices"][0]["delta"];
            let mut got = false;
            if let Some(s) = delta["reasoning_content"].as_str()
                && !s.is_empty()
            {
                reasoning.push_str(s);
                got = true;
            }
            if let Some(s) = delta["content"].as_str()
                && !s.is_empty()
            {
                content.push_str(s);
                got = true;
            }
            if got {
                generated += 1;
                kv.generated.store(generated, Ordering::Relaxed);
                if ttft.is_none() {
                    ttft = Some(started.elapsed().as_millis());
                }
            }

            if last_report.elapsed() >= Duration::from_millis(100) {
                last_report = Instant::now();
                on_progress(snapshot(started, generated, ttft, &kv));
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    let _ = poller.await;

    let mut progress = snapshot(started, generated, ttft, &kv);
    if usage_prompt > 0 {
        progress.prompt_tokens = usage_prompt;
        progress.kv_used = usage_prompt + generated as u32;
    }
    on_progress(progress);

    Ok(ChatResult {
        content,
        prompt_tokens: usage_prompt,
        completion_tokens: if usage_completion > 0 {
            usage_completion
        } else {
            generated
        },
        ttft_ms: ttft,
        total_ms: started.elapsed().as_millis(),
    })
}
