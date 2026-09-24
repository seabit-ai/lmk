#!/bin/bash
set -u
cd "$(dirname "$0")"
lmk down
~/src/lmk/.venv/bin/python run_mtp.py raw "$@" 2>&1 | tee run.log
lmk up >/dev/null && lmk status | head -3
