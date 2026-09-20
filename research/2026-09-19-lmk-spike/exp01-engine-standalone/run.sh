#!/bin/sh
# exp01: 借 LM Studio 自带的 Python 与库，脱离它的 server 直接驱动开源 mlx-engine。
cd "$(dirname "$0")"
V="$HOME/.lmstudio/extensions/backends/vendor/_amphibian"
PY=$(ls $V/cpython3.11-mac-arm64@*/bin/python3 | tail -1)
SP="$V/app-mlx-generate-mac14-arm64@34/lib/python3.11/site-packages"
MODEL="$HOME/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"
PYTHONPATH="$SP" "$PY" spike.py "$MODEL" > result.jsonl 2> stderr.log
echo "exit=$?" >> result.jsonl
