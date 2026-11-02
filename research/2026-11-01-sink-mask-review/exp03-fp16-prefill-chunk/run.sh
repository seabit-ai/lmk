#!/bin/bash
# exp03: the fp16-cache arm at chunk 512 (exp02 measured fp16 only at 2048). Same bench as exp02, kept in ../shared.
set -u
X=$(cd "$(dirname "$0")" && pwd); L=$X/../../..; E=$L/.engine/mlx-engine
cd "$X" || exit 1
echo "engine $(git -C "$E" rev-parse --short HEAD) lmk $(git -C "$L" rev-parse --short HEAD) $(date)" | tee raw/run.log
echo "=== kv16 step512 $(date +%T)" | tee -a raw/run.log
PYTHONPATH="$E:$L" "$L/.venv/bin/python" ../shared/bench.py --kv-bits 16 --step 512 --out raw/kv16-step512.jsonl >raw/kv16-step512.log 2>&1
echo "exit $? $(date +%T)" | tee -a raw/run.log
echo "done $(date)" | tee -a raw/run.log
