#!/bin/bash
# exp13: stop the resident service (GPU to the experiment only), run A then B, bring it back.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LMK="$(cd "$HERE/../../.." && pwd)"
PY="$LMK/.venv/bin/python"
~/.local/bin/lmk down
trap '~/.local/bin/lmk up' EXIT
"$PY" "$HERE/curve.py" "$HERE/raw" 2>&1 | tee "$HERE/raw/curve.log"
PYTHONPATH="$LMK/.engine/mlx-engine:$LMK" "$PY" "$HERE/rounds.py" "$HERE/raw" 2>&1 | tee "$HERE/raw/rounds.log"
