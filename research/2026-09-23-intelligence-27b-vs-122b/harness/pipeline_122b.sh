#!/bin/bash
set -u
S="$1"; cd ~/src/lmk; export PYTHONPATH=$PWD/.engine/mlx-engine:$PWD
D=research/2026-09-23-intelligence-27b-vs-122b
~/.local/bin/lmk down >/dev/null 2>&1; sleep 3; echo "$(date +%H:%M) resident stopped"
H="$S/eval-122B-off"; mkdir -p "$H/logs"
printf 'model: {name: qwen3.5-122b-a10b-4bit, thinking: false}\nlisten: {port: 1236}\ncache: {dir: "%s/cache"}\nlog: {dir: "%s/logs"}\n' "$H" "$H" > "$H/config.yaml"
LMK_HOME="$H" .venv/bin/python -P -m lmk serve > "$S/eval-122B-off.log" 2>&1 & echo $! > "$S/eval-122B-off.pid"
until curl -s -m 2 http://127.0.0.1:1236/lmk/v1/status >/dev/null; do sleep 3; kill -0 "$(cat "$S/eval-122B-off.pid")" 2>/dev/null || { echo "server died"; break; }; done
echo "$(date +%H:%M) 122B-off server up"
.venv/bin/python "$D/harness/run.py" 122B-off http://127.0.0.1:1236 qwen3.5-122b-a10b-4bit > "$S/eval-122B-off.progress" 2>&1
echo "$(date +%H:%M) 122B-off done: $(grep -c ': PASS' "$S/eval-122B-off.progress") pass / $(grep -c ': fail' "$S/eval-122B-off.progress") fail"
kill "$(cat "$S/eval-122B-off.pid")" 2>/dev/null; wait "$(cat "$S/eval-122B-off.pid")" 2>/dev/null; sleep 3
~/.local/bin/lmk up 2>&1 | grep -E "lmk is up|✗"; echo "$(date +%H:%M) 122B ARM DONE"
