#!/usr/bin/env bash
# 看门狗：每 30 秒检查 PostgreSQL / 嵌入 / 聊天（可选精排），掉了就拉起。
# 用法：./scripts/supervise.sh [--rerank]
set -euo pipefail
cd "$(dirname "$0")/.."

export NOTE_EMBED="${NOTE_EMBED:-http}"
export NOTE_EMBED_URL="${NOTE_EMBED_URL:-http://127.0.0.1:8081/v1}"
export NOTE_CHAT_URL="${NOTE_CHAT_URL:-http://127.0.0.1:8082/v1}"
export NOTE_RERANK_URL="${NOTE_RERANK_URL:-http://127.0.0.1:8083/rerank}"

INTERVAL="${SUPERVISE_INTERVAL:-30}"

# --watch 自身是死循环；外层只兜底进程崩溃。
while true; do
  cargo run -q -- supervise --watch --interval "$INTERVAL" "$@" || true
  sleep 5
done
