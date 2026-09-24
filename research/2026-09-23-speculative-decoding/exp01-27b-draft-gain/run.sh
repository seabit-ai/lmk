#!/bin/bash
# exp01: stop the resident lmk, measure, bring it back. Output: raw/*.json + run.log
set -u
cd "$(dirname "$0")"
lmk down
~/src/lmk/.venv/bin/python run.py raw "$@" 2>&1 | tee run.log
lmk up
