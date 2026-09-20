#!/bin/sh
# exp03: exp01 的同一份 spike.py，换成干净环境（公开仓库的 mlx_engine + pip 装的依赖）。
cd "$(dirname "$0")"
: "${LMK_SPIKE_DIR:?set LMK_SPIKE_DIR}"
MODEL="$HOME/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"
cp "$LMK_SPIKE_DIR/engine-commit.txt" "$LMK_SPIKE_DIR/pip-freeze.txt" . 2>/dev/null
PYTHONPATH="$LMK_SPIKE_DIR/mlx-engine" "$LMK_SPIKE_DIR/venv/bin/python" spike.py "$MODEL" > result.jsonl 2> stderr.log
echo "exit=$?" >> result.jsonl
