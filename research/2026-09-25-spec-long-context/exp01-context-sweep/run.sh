#!/bin/bash
# exp01: speculative decoding (dflash2) on/off across 8k / 32k / 64k / 128k of context, qwen3.8-27b-4bit, kv8, thinking off.
# usage: research/2026-09-25-spec-long-context/exp01-context-sweep/run.sh
# Needs the lmk checkout's .venv and .engine at ENGINE_COMMIT, the model and the dflash2 draft in the HF cache.
# Leaves the resident service (port 1235) alone; runs a temporary lmk on 1236 per condition, killed by PID.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
RAW="$HERE/raw"; mkdir -p "$RAW"
CTXS="8192 32768 65536 131072"
PORT=1236
WORK="$(mktemp -d /tmp/lmk-slc.XXXXXX)"   # temporary LMK_HOMEs + one prompt cache shared by both conditions
PID=""
export PYTHONPATH="$ROOT/.engine/mlx-engine:$ROOT"

cleanup() {
  if [ -n "$PID" ]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi
  rm -rf "$WORK"
}
trap cleanup EXIT

status() { curl -s -m 5 "http://127.0.0.1:$1/lmk/v1/status"; }

MODEL_DIR="$(status 1235 | python3 -c 'import sys,json; print(json.load(sys.stdin)["model"]["path"])')"
lmk status > "$RAW/resident-before.txt" 2>&1

# prefixes: public source of mlx-engine, mlx_lm, mlx_vlm, cut at exact token counts
.venv/bin/python "$HERE/prefix.py" "$MODEL_DIR" "$WORK/prefix" $CTXS | tee "$RAW/prefix.txt"

# one temporary server per condition, one after the other (never two at once)
for COND in off on; do
  H="$WORK/$COND"; mkdir -p "$H/logs"
  if [ "$COND" = on ]; then SPEC="speculative_decoding: true, draft: dflash2"; else SPEC="speculative_decoding: false"; fi
  cat > "$H/config.yaml" <<YAML
model: {name: qwen3.8-27b-4bit, kv_cache_bits: 8, thinking: false, $SPEC}
listen: {port: $PORT}
cache: {dir: "$WORK/cache"}
log: {dir: "$H/logs"}
YAML
  cp "$H/config.yaml" "$RAW/config-$COND.yaml"
  echo "$(date '+%F %T') start $COND" | tee -a "$RAW/timeline.txt"
  LMK_HOME="$H" .venv/bin/python -P -m lmk serve > "$RAW/serve-$COND.log" 2>&1 &
  PID=$!
  until status $PORT > /dev/null; do sleep 2; kill -0 "$PID" || { echo "serve died"; exit 1; }; done
  status $PORT > "$RAW/status-$COND-start.json"
  .venv/bin/python "$HERE/measure.py" "http://127.0.0.1:$PORT" "$H/logs/lmk.jsonl" "$WORK/prefix" "$COND" \
    "$RAW/runs.jsonl" $CTXS | tee "$RAW/progress-$COND.txt"
  status $PORT > "$RAW/status-$COND-end.json"
  kill "$PID"; wait "$PID" 2>/dev/null || true; PID=""
  cp "$H/logs/lmk.jsonl" "$RAW/lmk-$COND.jsonl"
  echo "$(date '+%F %T') stop $COND" | tee -a "$RAW/timeline.txt"
done
lmk status > "$RAW/resident-after.txt" 2>&1

.venv/bin/python "$HERE/table.py" "$RAW/runs.jsonl" | tee "$HERE/results.md"
