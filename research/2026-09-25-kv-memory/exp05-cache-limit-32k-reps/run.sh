#!/bin/bash
# exp05: 32k decode speed with the MLX cache limit unset / 1 GiB / 4 GiB, kv8 and kv16, conditions alternated
# twice (exp02/exp04's 32k rows were too noisy to tell a 1 GiB slowdown). qwen3.8-27b-4bit, dflash2 draft, code task,
# 5 x 256-token decodes per condition, one temporary lmk per condition.
# usage: research/2026-09-25-kv-memory/exp05-cache-limit-32k-reps/run.sh
# Same harness as exp01/exp02 (copied); the limit is set by probe/sitecustomize.py from
# LMK_EXP_CACHE_LIMIT. Never runs make; port 1236, killed by PID.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
LMK_CHECKOUT="${LMK_CHECKOUT:-$HOME/src/lmk}"; export LMK_CHECKOUT
PY="$LMK_CHECKOUT/.venv/bin/python"
RAW="${RAW:-$HERE/raw}"; mkdir -p "$RAW"
CTXS="${CTXS:-32768}"
# name:kv bits:cache_limit_bytes — per precision: unset, 1 GiB, 4 GiB, then the same three again
CONDS="${CONDS:-kv8-unset-a:8: kv8-1g-a:8:1073741824 kv8-4g-a:8:4294967296 kv8-unset-b:8: kv8-1g-b:8:1073741824 kv8-4g-b:8:4294967296 kv16-unset-a:16: kv16-1g-a:16:1073741824 kv16-4g-a:16:4294967296 kv16-unset-b:16: kv16-1g-b:16:1073741824 kv16-4g-b:16:4294967296}"
PORT=1236
WORK="$(mktemp -d /tmp/lmk-kvm5.XXXXXX)"   # one prompt cache per precision: only its first condition prefills cold
PID=""
export PYTHONPATH="$HERE/probe:$LMK_CHECKOUT/.engine/mlx-engine:$ROOT"
export KINDS=code REPS=5 AFTER_S=3

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
  IFS=: read -r COND BITS LIMIT <<< "$SPEC"
  H="$WORK/$COND"; mkdir -p "$H/logs"
  FREE="$(free_gb)"
  echo "$(date '+%F %T') free before loading $COND: ${FREE} GB" | tee -a "$RAW/timeline.txt"
  [ "$FREE" -ge 45 ] || { echo "less than 45 GB free, not loading the model"; exit 1; }
  cat > "$H/config.yaml" <<YAML
model: {name: qwen3.8-27b-4bit, kv_cache_bits: $BITS, thinking: false, speculative_decoding: true, draft: dflash2}
listen: {port: $PORT}
cache: {dir: "$WORK/cache-kv$BITS"}
log: {dir: "$H/logs"}
YAML
  cp "$H/config.yaml" "$RAW/config-$COND.yaml"
  echo "$(date '+%F %T') start $COND (kv $BITS, cache limit '${LIMIT}')" | tee -a "$RAW/timeline.txt"
  LMK_HOME="$H" LMK_EXP_PROBE="$RAW/probe-$COND.jsonl" LMK_EXP_CACHE_LIMIT="$LIMIT" \
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
