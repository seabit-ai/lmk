#!/bin/bash
# exp03: per-Mac-size table from lmk's formulas + the lines measured in exp01 (uncapped) and exp02 / exp06 (4 GiB cap).
# No model is loaded. usage: research/2026-09-25-kv-memory/exp03-per-mac-size/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
LMK_CHECKOUT="${LMK_CHECKOUT:-$HOME/src/lmk}"
PYTHONPATH="$ROOT" "$LMK_CHECKOUT/.venv/bin/python" -P "$HERE/compute.py" \
  "$HERE/../exp01-kv8-vs-kv16-memory/raw/runs.jsonl" "$HERE/../exp02-cache-limit/raw/runs.jsonl" \
  "$HERE/../exp06-kv8-4g-limit/raw/runs.jsonl" $((4 << 30)) | tee "$HERE/results.md"
