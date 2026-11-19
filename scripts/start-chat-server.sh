#!/usr/bin/env bash
# 起本地聊天模型（Qwen3-VL-8B-Instruct Q4_K_M）。
# 8 GB 卡上和嵌入服务（0.6B）同时驻留时，KV 放内存，权重留在显存。
set -euo pipefail

BIN="${LLAMA_SERVER:-/tools/llama.cpp-prism/llama-prism-b12128-d7b2896/llama-server}"
MODEL="${CHAT_MODEL_PATH:-/models/lmstudio-community/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q4_K_M.gguf}"
PORT="${CHAT_PORT:-8082}"
CTX="${CHAT_CTX:-4096}"

exec "$BIN" \
  -m "$MODEL" \
  -ngl 99 \
  -c "$CTX" \
  -b 2048 \
  -ub 1024 \
  --no-kv-offload \
  --slots \
  --metrics \
  --host 127.0.0.1 \
  --port "$PORT" \
  --no-webui
