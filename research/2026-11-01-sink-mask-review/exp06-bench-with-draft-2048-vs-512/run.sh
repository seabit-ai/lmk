#!/bin/bash
# exp06: `lmk bench` (the user's own yardstick, draft model on) against a temporary lmk, engine at chunk 2048 (c3ee66f)
# and at chunk 512 (e863778). Same config, same bench seed, fresh cache dir per arm, one arm at a time, port 1299.
set -u
X=$(cd "$(dirname "$0")" && pwd); L=$X/../../..; cd "$X" || exit 1
MODEL=$(ls -d ~/.cache/huggingface/hub/models--lmstudio-community--Qwen3.8-27B-MLX-4bit/snapshots/*/ | tail -1)
SEED=20261101
echo "model $MODEL seed $SEED $(date)" | tee raw/run.log
for arm in "2048 c3ee66f $L/.engine/mlx-engine-spec" "512 e863778 $L/.engine/mlx-engine"; do
  set -- $arm; CHUNK=$1; COMMIT=$2; ENGINE=$3
  git -C "$ENGINE" checkout -q "$COMMIT" || { echo "checkout $COMMIT failed in $ENGINE" | tee -a raw/run.log; exit 1; }
  HOME_T=$(mktemp -d /tmp/lmk-exp06-XXXX); mkdir -p "$HOME_T/cache" "$HOME_T/logs"
  ln -s ~/.lmk/drafts "$HOME_T/drafts"   # reuse a built MTP head if the key matches, else lmk builds one here
  cat > "$HOME_T/config.yaml" <<CFG
model:
  path: $MODEL
  reasoning_effort: low
  kv_cache_bits: 16
  speculative_decoding: true
  draft_weights: official
listen:
  host: 127.0.0.1
  port: 1299
cache:
  dir: $HOME_T/cache
  max_size: 20G
requests:
  max_parallel: 2
  max_queue: 16
  max_wait_seconds: 600
CFG
  echo "=== chunk $CHUNK engine $(git -C "$ENGINE" rev-parse --short HEAD) home $HOME_T $(date +%T)" | tee -a raw/run.log
  LMK_HOME=$HOME_T PYTHONPATH="$ENGINE:$L" "$L/.venv/bin/python" -P -m lmk serve >"raw/serve-$CHUNK.log" 2>&1 &
  PID=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:1299/v1/models >/dev/null 2>&1 && break; sleep 2; done
  curl -sf http://127.0.0.1:1299/v1/models >/dev/null || { echo "serve did not come up (pid $PID)" | tee -a raw/run.log; kill $PID; exit 1; }
  echo "up after $((i*2))s $(date +%T)" | tee -a raw/run.log
  LMK_HOME=$HOME_T PYTHONPATH="$ENGINE:$L" "$L/.venv/bin/python" -P -m lmk bench --url http://127.0.0.1:1299 --seed $SEED >"raw/bench-$CHUNK.txt" 2>&1
  echo "bench exit $? $(date +%T)" | tee -a raw/run.log
  LMK_HOME=$HOME_T PYTHONPATH="$ENGINE:$L" "$L/.venv/bin/python" -P -m lmk status >"raw/status-$CHUNK.txt" 2>&1
  kill $PID; wait $PID 2>/dev/null; echo "stopped pid $PID $(date +%T)" | tee -a raw/run.log
done
git -C "$L/.engine/mlx-engine-spec" checkout -q lmk-spec; git -C "$L/.engine/mlx-engine" checkout -q lmk
echo "done $(date)" | tee -a raw/run.log
