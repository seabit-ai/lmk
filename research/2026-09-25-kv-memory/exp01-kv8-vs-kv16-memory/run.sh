#!/bin/bash
# exp01: memory of qwen3.8-27b-4bit + dflash2 draft at kv_cache_bits 8 vs 16, 8k / 32k / 64k / 128k context,
# decode 256 tokens after a cached prefix. Per request: ps resident bytes, MLX active / cache / peak (probe).
# usage: research/2026-09-25-kv-memory/exp01-kv8-vs-kv16-memory/run.sh
# Runs the lmk code of the checkout this file is in, with the Python env and engine of LMK_CHECKOUT
# (default: the main checkout /Users/<you>/src/lmk — a worktree has no .venv/.engine). Never runs make.
# One temporary lmk on port 1236 at a time, killed by PID; leaves port 1235 alone.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
LMK_CHECKOUT="${LMK_CHECKOUT:-$HOME/src/lmk}"; export LMK_CHECKOUT
PY="$LMK_CHECKOUT/.venv/bin/python"
RAW="${RAW:-$HERE/raw}"; mkdir -p "$RAW"   # RAW / CTXS / BITS_LIST overridable for a smoke run
CTXS="${CTXS:-8192 32768 65536 131072}"
PORT=1236
WORK="$(mktemp -d /tmp/lmk-kvm1.XXXXXX)"
PID=""
# probe/ first: its sitecustomize.py is imported at interpreter start
export PYTHONPATH="$HERE/probe:$LMK_CHECKOUT/.engine/mlx-engine:$ROOT"

cleanup() {
  if [ -n "$PID" ]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi
  rm -rf "$WORK"
}
trap cleanup EXIT

status() { curl -s -m 5 "http://127.0.0.1:$1/lmk/v1/status"; }
free_gb() { "$PY" -P -c 'from lmk.memory import SystemMemory as S; m=S().read(); print(int(m.total_bytes*m.free_percent/100/1e9))'; }

echo "$(date '+%F %T') engine $(git -C "$LMK_CHECKOUT/.engine/mlx-engine" rev-parse --short HEAD), lmk $(git -C "$ROOT" rev-parse --short HEAD), resident on 1235: $(status 1235 > /dev/null && echo up || echo down)" | tee -a "$RAW/timeline.txt"

MODEL_DIR="$(ls -d "$HOME"/.cache/huggingface/hub/models--lmstudio-community--Qwen3.8-27B-MLX-4bit/snapshots/*/ | head -1)"
"$PY" "$HERE/prefix.py" "$MODEL_DIR" "$WORK/prefix" $CTXS | tee "$RAW/prefix.txt"

for BITS in ${BITS_LIST:-8 16}; do
  COND="kv$BITS"
  H="$WORK/$COND"; mkdir -p "$H/logs"
  FREE="$(free_gb)"
  echo "$(date '+%F %T') free before loading $COND: ${FREE} GB" | tee -a "$RAW/timeline.txt"
  [ "$FREE" -ge 45 ] || { echo "less than 45 GB free, not loading the model"; exit 1; }
  cat > "$H/config.yaml" <<YAML
model: {name: qwen3.8-27b-4bit, kv_cache_bits: $BITS, thinking: false, speculative_decoding: true, draft: dflash2}
listen: {port: $PORT}
cache: {dir: "$WORK/cache-$COND"}
log: {dir: "$H/logs"}
YAML
  cp "$H/config.yaml" "$RAW/config-$COND.yaml"
  echo "$(date '+%F %T') start $COND" | tee -a "$RAW/timeline.txt"
  LMK_HOME="$H" LMK_EXP_PROBE="$RAW/probe-$COND.jsonl" "$PY" -P -m lmk serve > "$RAW/serve-$COND.log" 2>&1 &
  PID=$!
  until status $PORT > /dev/null; do sleep 2; kill -0 "$PID" || { echo "serve died"; exit 1; }; done
  status $PORT > "$RAW/status-$COND-start.json"
  grep -h "context auto-fit\|context target" "$RAW/serve-$COND.log" > "$RAW/context-fit-$COND.txt" || true
  "$PY" -P "$HERE/measure.py" "http://127.0.0.1:$PORT" "$H/logs/lmk.jsonl" "$RAW/probe-$COND.jsonl" "$PID" "$WORK/prefix" "$COND" \
    "$RAW/runs.jsonl" $CTXS | tee "$RAW/progress-$COND.txt"
  status $PORT > "$RAW/status-$COND-end.json"
  kill "$PID"; wait "$PID" 2>/dev/null || true; PID=""
  cp "$H/logs/lmk.jsonl" "$RAW/lmk-$COND.jsonl"
  echo "$(date '+%F %T') stop $COND" | tee -a "$RAW/timeline.txt"
done

# public repo: no home paths in raw
grep -rl "$HOME" "$RAW" | while read -r f; do sed -i '' "s|$HOME|~|g" "$f"; done
"$PY" -P "$HERE/table.py" "$RAW/runs.jsonl" | tee "$HERE/results.md"
