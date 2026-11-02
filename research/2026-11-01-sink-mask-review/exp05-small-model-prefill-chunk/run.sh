#!/bin/bash
# exp05: the smallest catalog model (gemma-4-E4B-it 4-bit) at chunk 2048 and 512, fp16 cache. One process per arm.
set -u
X=$(cd "$(dirname "$0")" && pwd); L=$X/../../..; E=$L/.engine/mlx-engine
M=$(ls -d ~/.cache/huggingface/hub/models--lmstudio-community--gemma-4-E4B-it-MLX-4bit/snapshots/*/ | tail -1)
cd "$X" || exit 1
echo "engine $(git -C "$E" rev-parse --short HEAD) lmk $(git -C "$L" rev-parse --short HEAD) model $M $(date)" | tee raw/run.log
for step in 2048 512; do
  echo "=== e4b kv16 step$step $(date +%T)" | tee -a raw/run.log
  PYTHONPATH="$E:$L" "$L/.venv/bin/python" ../shared/bench_model.py --model "$M" --name e4b --kv-bits 16 --step "$step" --out "raw/e4b-kv16-step$step.jsonl" >"raw/e4b-kv16-step$step.log" 2>&1
  echo "exit $? $(date +%T)" | tee -a raw/run.log
done
echo "done $(date)" | tee -a raw/run.log
