#!/bin/sh
# exp02: 让模型写一个工具调用，留下原始文本，再喂给 mlx-lm 自带的 qwen3_coder 解析器。
cd "$(dirname "$0")"
V="$HOME/.lmstudio/extensions/backends/vendor/_amphibian"
PY=$(ls $V/cpython3.11-mac-arm64@*/bin/python3 | tail -1)
SP="$V/app-mlx-generate-mac14-arm64@34/lib/python3.11/site-packages"
MODEL="$HOME/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit"
PYTHONPATH="$SP" "$PY" spike.py "$MODEL" > result.json 2> stderr.log
echo "exit=$?" >> result.json
