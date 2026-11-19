#!/usr/bin/env bash
# 用本机 llama.cpp 起一个 OpenAI 兼容的嵌入服务（Qwen3-Embedding-0.6B, 1024 维）。
# 需要换模型时改 EMBED_MODEL_PATH / EMBED_PORT 即可。
set -euo pipefail

BIN="${LLAMA_SERVER:-/tools/llama.cpp-prism/llama-prism-b12128-d7b2896/llama-server}"
MODEL="${EMBED_MODEL_PATH:-/models/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf}"
PORT="${EMBED_PORT:-8081}"

exec "$BIN" \
  -m "$MODEL" \
  --embedding \
  -ngl 99 \
  -c 4096 \
  -b 2048 \
  -ub 1024 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --no-warmup
