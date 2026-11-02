#!/bin/bash
# exp04: A decodes while B reads a 30k prompt; chunk 2048 vs 512, fp16 cache (the owner's configuration). One process per arm.
set -u
X=$(cd "$(dirname "$0")" && pwd); L=$X/../../..; E=$L/.engine/mlx-engine
cd "$X" || exit 1
echo "engine $(git -C "$E" rev-parse --short HEAD) lmk $(git -C "$L" rev-parse --short HEAD) $(date)" | tee raw/run.log
for step in 2048 512; do
  echo "=== kv16 step$step B30000 $(date +%T)" | tee -a raw/run.log
  PYTHONPATH="$E:$L" "$L/.venv/bin/python" ../shared/bench_stall.py --kv-bits 16 --step "$step" --prompt-tokens 30000 --out "raw/kv16-step$step-B30000.jsonl" >"raw/kv16-step$step-B30000.log" 2>&1
  echo "exit $? $(date +%T)" | tee -a raw/run.log
done
echo "done $(date)" | tee -a raw/run.log
