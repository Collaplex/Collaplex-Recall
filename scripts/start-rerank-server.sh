#!/usr/bin/env bash
# 交叉编码器精排服务（bge-reranker-v2-m3 Q4_K_M）。
# 与 0.6B 嵌入、8B 聊天同时驻留时约 7.1 GB 显存；不够就加 -ngl 0 走 CPU。
set -euo pipefail

BIN="${LLAMA_SERVER:-/tools/llama.cpp-prism/llama-prism-b12128-d7b2896/llama-server}"
MODEL="${RERANK_MODEL_PATH:-/models/gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q4_K_M.gguf}"
PORT="${RERANK_PORT:-8083}"
NGL="${RERANK_NGL:-99}"

exec "$BIN" \
  -m "$MODEL" \
  --reranking \
  -ngl "$NGL" \
  -c 512 \
  -b 512 \
  -ub 512 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --no-webui
