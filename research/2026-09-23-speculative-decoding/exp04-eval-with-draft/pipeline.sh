#!/bin/bash
set -u
S="$1"; cd ~/src/lmk; export PYTHONPATH=$PWD/.engine/mlx-engine:$PWD
E=research/2026-09-23-speculative-decoding/exp04-eval-with-draft; H27=research/2026-09-23-intelligence-27b-vs-122b/harness
export EVAL_RAW_DIR=$PWD/$E/raw EVAL_RUNS=2 EVAL_LIMIT=20
H="$S/spec-eval"; mkdir -p "$H/logs"
printf 'model: {name: qwen3.8-27b-4bit, reasoning_effort: low, speculative_decoding: true}\nlisten: {port: 1236}\ncache: {dir: "%s/cache"}\nlog: {dir: "%s/logs"}\n' "$H" "$H" > "$H/config.yaml"
lmk down >/dev/null
LMK_HOME="$H" .venv/bin/python -P -m lmk serve > "$S/spec-eval.log" 2>&1 & echo $! > "$S/spec-eval.pid"
until curl -s -m 2 http://127.0.0.1:1236/lmk/v1/status >/dev/null; do sleep 2; kill -0 "$(cat "$S/spec-eval.pid")" 2>/dev/null || { echo "server died"; exit 1; }; done
echo "$(date +%H:%M) server up"
.venv/bin/python "$H27/run.py" draft-on http://127.0.0.1:1236 qwen3.8-27b-4bit code instruct tools > "$S/spec-eval.progress" 2>&1
echo "$(date +%H:%M) done: $(grep -c PASS "$S/spec-eval.progress") pass / $(grep -c ': fail' "$S/spec-eval.progress") fail"
curl -s http://127.0.0.1:1236/lmk/v1/status | python3 -c "import sys,json; print('draft stats', json.load(sys.stdin).get('draft'))" | tee "$E/draft-stats.txt"
kill "$(cat "$S/spec-eval.pid")"; wait "$(cat "$S/spec-eval.pid")" 2>/dev/null; sleep 3
lmk up >/dev/null; echo "$(date +%H:%M) ALL DONE, resident back"
EVAL_RAW_DIR=$PWD/$E/raw .venv/bin/python "$H27/run.py" grade | tee "$E/grade.md"
