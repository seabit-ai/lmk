#!/bin/bash
# exp02: exp01 with kv_cache_bits 16 — speculative decoding (dflash2) on/off across 8k / 32k / 64k / 128k, qwen3.8-27b-4bit, KV at the model's own precision, thinking off.
# usage: research/2026-09-25-spec-long-context/exp02-kv16-context-sweep/run.sh
# Needs the lmk checkout's .venv and .engine at ENGINE_COMMIT, the model and the dflash2 draft in the HF cache.
# Leaves the resident service (port 1235) as it is (up or down; measure.py discards runs that overlap its activity); runs a temporary lmk on 1236 per condition, killed by PID.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
RAW="$HERE/raw"; mkdir -p "$RAW"
CTXS="8192 32768 65536 131072"
PORT=1236
WORK="$(mktemp -d /tmp/lmk-slc2.XXXXXX)"   # temporary LMK_HOMEs + one prompt cache shared by both conditions
PID=""
export PYTHONPATH="$ROOT/.engine/mlx-engine:$ROOT"

cleanup() {
  if [ -n "$PID" ]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi
  rm -rf "$WORK"
}
trap cleanup EXIT

status() { curl -s -m 5 "http://127.0.0.1:$1/lmk/v1/status"; }

# memory check before loading the model (~17 GB + up to ~9 GB of 16-bit KV at 128k + prefill buffers);
# the resident service may be up (then it holds ~17 GB more) or down
FREE_GB="$(.venv/bin/python -P -c 'from lmk.memory import SystemMemory as S; m=S().read(); print(int(m.total_bytes*m.free_percent/100/1e9))')"
echo "$(date '+%F %T') free before loading: ${FREE_GB} GB" | tee -a "$RAW/timeline.txt"
[ "$FREE_GB" -ge 45 ] || { echo "less than 45 GB free, not loading the model"; exit 1; }

MODEL_DIR="$(ls -d "$HOME"/.cache/huggingface/hub/models--lmstudio-community--Qwen3.8-27B-MLX-4bit/snapshots/*/ | head -1)"
{ lmk status 2>&1 || true; } | sed -n 1,10p > "$RAW/resident-before.txt"   # header only: "just finished" names the owner's sessions

# prefixes: public source of mlx-engine, mlx_lm, mlx_vlm, cut at exact token counts
.venv/bin/python "$HERE/prefix.py" "$MODEL_DIR" "$WORK/prefix" $CTXS | tee "$RAW/prefix.txt"

# one temporary server per condition, one after the other (never two at once)
for COND in off on; do
  H="$WORK/$COND"; mkdir -p "$H/logs"
  if [ "$COND" = on ]; then SPEC="speculative_decoding: true, draft: dflash2"; else SPEC="speculative_decoding: false"; fi
  cat > "$H/config.yaml" <<YAML
model: {name: qwen3.8-27b-4bit, kv_cache_bits: 16, thinking: false, $SPEC}
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
  grep -h "context auto-fit\|context target" "$RAW/serve-$COND.log" > "$RAW/context-fit-$COND.txt" || true
  .venv/bin/python "$HERE/measure.py" "http://127.0.0.1:$PORT" "$H/logs/lmk.jsonl" "$WORK/prefix" "$COND" \
    "$RAW/runs.jsonl" $CTXS | tee "$RAW/progress-$COND.txt"
  status $PORT > "$RAW/status-$COND-end.json"
  kill "$PID"; wait "$PID" 2>/dev/null || true; PID=""
  cp "$H/logs/lmk.jsonl" "$RAW/lmk-$COND.jsonl"
  echo "$(date '+%F %T') stop $COND" | tee -a "$RAW/timeline.txt"
done
{ lmk status 2>&1 || true; } | sed -n 1,10p > "$RAW/resident-after.txt"

# public repo: no home paths in raw
grep -rl "$HOME" "$RAW" | while read -r f; do sed -i '' "s|$HOME|~|g" "$f"; done
.venv/bin/python "$HERE/table.py" "$HERE/../exp01-context-sweep/raw/runs.jsonl" "$RAW/runs.jsonl" | tee "$HERE/results.md"
