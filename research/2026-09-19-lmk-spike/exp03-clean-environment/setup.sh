#!/bin/sh
# exp03 setup: 从公开仓库装一套干净环境（不借 LM Studio 的任何东西）。装在仓库之外的 $LMK_SPIKE_DIR。
set -e
: "${LMK_SPIKE_DIR:?set LMK_SPIKE_DIR to a scratch directory outside the repo}"
mkdir -p "$LMK_SPIKE_DIR" && cd "$LMK_SPIKE_DIR"
[ -d mlx-engine ] || git clone --depth 1 https://github.com/lmstudio-ai/mlx-engine.git
git -C mlx-engine rev-parse HEAD > engine-commit.txt
[ -d venv ] || "$HOME/.local/bin/python3.11" -m venv venv
./venv/bin/pip install --quiet --upgrade pip
./venv/bin/pip install --quiet -r mlx-engine/requirements.txt
./venv/bin/pip freeze > pip-freeze.txt
echo SETUP-DONE
