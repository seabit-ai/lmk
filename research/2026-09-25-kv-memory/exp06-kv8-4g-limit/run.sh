#!/bin/bash
# exp06: exp04 with a 4 GiB MLX cache limit (exp05: 1 GiB slows the kv8 decode at 32k by ~7%, 4 GiB does not).
# qwen3.8-27b-4bit, dflash2 draft, 32k and 128k, code task, 3 x 256-token decodes per context, one temporary lmk per condition.
# usage: research/2026-09-25-kv-memory/exp06-kv8-4g-limit/run.sh
# Same harness as exp01/exp02 (copied); the limit is set by probe/sitecustomize.py from
# LMK_EXP_CACHE_LIMIT, clear-at-end by LMK_EXP_CLEAR_AT_END. Never runs make; port 1236, killed by PID.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
LMK_CHECKOUT="${LMK_CHECKOUT:-$HOME/src/lmk}"; export LMK_CHECKOUT
PY="$LMK_CHECKOUT/.venv/bin/python"
RAW="${RAW:-$HERE/raw}"; mkdir -p "$RAW"
CTXS="${CTXS:-32768 131072}"
# name:cache_limit_bytes:clear_at_end — the unlimited baseline first and again last (drift check)
CONDS="${CONDS:-unset:: limit4g:4294967296:}"
PORT=1236
WORK="$(mktemp -d /tmp/lmk-kvm6.XXXXXX)"   # one prompt cache shared by all conditions: only the first prefills cold
PID=""
export PYTHONPATH="$HERE/probe:$LMK_CHECKOUT/.engine/mlx-engine:$ROOT"
export KINDS=code

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

for SPEC in $CONDS; do
  IFS=: read -r COND LIMIT CLEAR <<< "$SPEC"
  H="$WORK/$COND"; mkdir -p "$H/logs"
  FREE="$(free_gb)"
  echo "$(date '+%F %T') free before loading $COND: ${FREE} GB" | tee -a "$RAW/timeline.txt"
  [ "$FREE" -ge 45 ] || { echo "less than 45 GB free, not loading the model"; exit 1; }
  cat > "$H/config.yaml" <<YAML
model: {name: qwen3.8-27b-4bit, kv_cache_bits: 8, thinking: false, speculative_decoding: true, draft: dflash2}
listen: {port: $PORT}
cache: {dir: "$WORK/cache"}
log: {dir: "$H/logs"}
YAML
  cp "$H/config.yaml" "$RAW/config-$COND.yaml"
  echo "$(date '+%F %T') start $COND (cache limit '${LIMIT}', clear at end '${CLEAR}')" | tee -a "$RAW/timeline.txt"
  LMK_HOME="$H" LMK_EXP_PROBE="$RAW/probe-$COND.jsonl" LMK_EXP_CACHE_LIMIT="$LIMIT" LMK_EXP_CLEAR_AT_END="$CLEAR" \
    "$PY" -P -m lmk serve > "$RAW/serve-$COND.log" 2>&1 &
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

grep -rl "$HOME" "$RAW" | while read -r f; do sed -i '' "s|$HOME|~|g" "$f"; done
"$PY" -P "$HERE/table.py" "$RAW/runs.jsonl" | tee "$HERE/results.md"
