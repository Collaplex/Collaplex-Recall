#!/usr/bin/env bash
# 带默认环境起 TUI。前缀必须和 ingest 时一致，否则向量会是 0 候选。
set -euo pipefail
cd "$(dirname "$0")/.."

export NOTE_EMBED="${NOTE_EMBED:-http}"
export NOTE_EMBED_URL="${NOTE_EMBED_URL:-http://127.0.0.1:8081/v1}"
export NOTE_EMBED_MODEL="${NOTE_EMBED_MODEL:-Qwen3-Embedding-0.6B}"
export NOTE_EMBED_DIM="${NOTE_EMBED_DIM:-1024}"
export NOTE_EMBED_QUERY_PREFIX="${NOTE_EMBED_QUERY_PREFIX:-$'Instruct: Given a user question, retrieve relevant note passages that answer it\nQuery: '}"
export NOTE_CHAT_URL="${NOTE_CHAT_URL:-http://127.0.0.1:8082/v1}"
export NOTE_CHAT_MODEL="${NOTE_CHAT_MODEL:-local-chat}"
export NOTE_CHAT_CTX="${NOTE_CHAT_CTX:-4096}"
export NOTE_REWRITE="${NOTE_REWRITE:-on}"
export NOTE_SELECT="${NOTE_SELECT:-on}"
export NOTE_CONTEXT_BUDGET="${NOTE_CONTEXT_BUDGET:-2000}"

exec cargo run -q -- tui
