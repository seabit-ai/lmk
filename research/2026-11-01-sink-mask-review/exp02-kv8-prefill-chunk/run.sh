#!/bin/bash
# exp02: four arms, one process each, sequential. Run from anywhere.
set -u
X=$(cd "$(dirname "$0")" && pwd); L=$X/../../..; E=$L/.engine/mlx-engine
cd "$X" || exit 1
echo "engine $(git -C "$E" rev-parse --short HEAD) lmk $(git -C "$L" rev-parse --short HEAD) $(date)" | tee raw/run.log
for arm in "16 2048" "8 2048" "8 1024" "8 512"; do
  set -- $arm
  echo "=== kv$1 step$2 $(date +%T)" | tee -a raw/run.log
  PYTHONPATH="$E:$L" "$L/.venv/bin/python" bench.py --kv-bits "$1" --step "$2" --out "raw/kv$1-step$2.jsonl" >"raw/kv$1-step$2.log" 2>&1
  echo "exit $? $(date +%T)" | tee -a raw/run.log
done
echo "done $(date)" | tee -a raw/run.log
