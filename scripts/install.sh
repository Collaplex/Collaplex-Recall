#!/usr/bin/env bash
# 安装 Collaplex Recall 到 ~/.local（PREFIX 可覆盖）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
LIB="$PREFIX/lib/collaplex-recall"
BINDIR="$PREFIX/bin"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/collaplex-recall"
LOGDIR="$DATA/logs"

echo "[1/4] 构建 release"
cargo build --release --locked --manifest-path "$ROOT/Cargo.toml"

echo "[2/4] 安装二进制与启动脚本到 $LIB"
install -Dm755 "$ROOT/target/release/recall" "$LIB/recall"
install -d "$LIB/scripts"
for s in start-embed-server.sh start-chat-server.sh start-rerank-server.sh; do
  install -m755 "$ROOT/scripts/$s" "$LIB/scripts/$s"
done
install -d "$LOGDIR"
# 首次安装时把源码目录里的历史日志带过去，不覆盖已有文件。
if [ -d "$ROOT/data/logs" ]; then
  cp -an "$ROOT/data/logs/." "$LOGDIR/" 2>/dev/null || true
fi

echo "[3/4] 写入 wrapper $BINDIR/recall"
mkdir -p "$BINDIR"
cat > "$BINDIR/recall" <<EOF
#!/usr/bin/env bash
# Collaplex Recall 启动器：带默认环境，可在任意目录运行。
set -euo pipefail
export NOTE_LOG_DIR="\${NOTE_LOG_DIR:-$LOGDIR}"
export NOTE_SCRIPT_DIR="\${NOTE_SCRIPT_DIR:-$LIB/scripts}"
export NOTE_EMBED="\${NOTE_EMBED:-http}"
export NOTE_EMBED_URL="\${NOTE_EMBED_URL:-http://127.0.0.1:8081/v1}"
export NOTE_EMBED_MODEL="\${NOTE_EMBED_MODEL:-Qwen3-Embedding-0.6B}"
export NOTE_EMBED_DIM="\${NOTE_EMBED_DIM:-1024}"
export NOTE_EMBED_QUERY_PREFIX="\${NOTE_EMBED_QUERY_PREFIX:-\$'Instruct: Given a user question, retrieve relevant note passages that answer it\\nQuery: '}"
export NOTE_CHAT_URL="\${NOTE_CHAT_URL:-http://127.0.0.1:8082/v1}"
export NOTE_CHAT_MODEL="\${NOTE_CHAT_MODEL:-local-chat}"
export NOTE_CHAT_CTX="\${NOTE_CHAT_CTX:-4096}"
export NOTE_CONTEXT_BUDGET="\${NOTE_CONTEXT_BUDGET:-2000}"
export NOTE_REWRITE="\${NOTE_REWRITE:-on}"
export NOTE_SELECT="\${NOTE_SELECT:-on}"
exec "$LIB/recall" "\$@"
EOF
chmod 755 "$BINDIR/recall"
ln -sfn recall "$BINDIR/collaplex-recall"

echo "[4/4] 完成"
echo "  recall --help"
echo "  recall tui"
echo "  recall supervise --watch --rerank"
