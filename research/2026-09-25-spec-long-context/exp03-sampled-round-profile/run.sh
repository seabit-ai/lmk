#!/bin/bash
# exp03: where a sampled speculative round's extra ~10 ms go — timers around the engine's DFlash round
# (monkeypatched in this process by profile_patch.py; lmk/ and the engine are not edited).
# qwen3.8-27b-4bit, kv8, dflash2, thinking off, 32k prefix (exp01's), prose + code, greedy vs default sampling.
# usage: research/2026-09-25-spec-long-context/exp03-sampled-round-profile/run.sh
# Leaves the resident service (port 1235) as it is; one temporary lmk on 1236, killed by PID.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
RAW="$HERE/raw"; mkdir -p "$RAW"
PORT=1236
WORK="$(mktemp -d /tmp/lmk-slc3.XXXXXX)"
PID=""
export PYTHONPATH="$ROOT/.engine/mlx-engine:$ROOT"
export LMK_PROF_CONTROL="$WORK/control" LMK_PROF_OUT="$RAW/rounds.jsonl"

cleanup() {
  if [ -n "$PID" ]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi
  rm -rf "$WORK"
}
trap cleanup EXIT

status() { curl -s -m 5 "http://127.0.0.1:$1/lmk/v1/status"; }

FREE_GB="$(.venv/bin/python -P -c 'from lmk.memory import SystemMemory as S; m=S().read(); print(int(m.total_bytes*m.free_percent/100/1e9))')"
echo "$(date '+%F %T') free before loading: ${FREE_GB} GB" | tee -a "$RAW/timeline.txt"
[ "$FREE_GB" -ge 40 ] || { echo "less than 40 GB free, not loading the model"; exit 1; }
{ lmk status 2>&1 || true; } | sed -n 1,10p > "$RAW/resident-before.txt"

MODEL_DIR="$(ls -d "$HOME"/.cache/huggingface/hub/models--lmstudio-community--Qwen3.8-27B-MLX-4bit/snapshots/*/ | head -1)"
.venv/bin/python "$HERE/../exp01-context-sweep/prefix.py" "$MODEL_DIR" "$WORK/prefix" 32768 | tee "$RAW/prefix.txt"

H="$WORK/home"; mkdir -p "$H/logs"
cat > "$H/config.yaml" <<YAML
model: {name: qwen3.8-27b-4bit, kv_cache_bits: 8, thinking: false, speculative_decoding: true, draft: dflash2}
listen: {port: $PORT}
cache: {dir: "$WORK/cache"}
log: {dir: "$H/logs"}
YAML
cp "$H/config.yaml" "$RAW/config.yaml"
echo "wall none" > "$LMK_PROF_CONTROL"
echo "$(date '+%F %T') start" | tee -a "$RAW/timeline.txt"
LMK_HOME="$H" .venv/bin/python -P "$HERE/profile_serve.py" > "$RAW/serve.log" 2>&1 &
PID=$!
until status $PORT > /dev/null; do sleep 2; kill -0 "$PID" || { echo "serve died"; exit 1; }; done
status $PORT > "$RAW/status-start.json"
.venv/bin/python "$HERE/measure.py" "http://127.0.0.1:$PORT" "$H/logs/lmk.jsonl" "$WORK/prefix/prefix-32768.txt" \
  "$LMK_PROF_CONTROL" "$RAW/runs.jsonl" | tee "$RAW/progress.txt"
status $PORT > "$RAW/status-end.json"
kill "$PID"; wait "$PID" 2>/dev/null || true; PID=""
cp "$H/logs/lmk.jsonl" "$RAW/lmk.jsonl"
echo "$(date '+%F %T') stop" | tee -a "$RAW/timeline.txt"
{ lmk status 2>&1 || true; } | sed -n 1,10p > "$RAW/resident-after.txt"

grep -rl "$HOME" "$RAW" | while read -r f; do sed -i '' "s|$HOME|~|g" "$f"; done
.venv/bin/python "$HERE/analyze.py" "$RAW/rounds.jsonl" "$RAW/runs.jsonl" | tee "$HERE/results.md"
