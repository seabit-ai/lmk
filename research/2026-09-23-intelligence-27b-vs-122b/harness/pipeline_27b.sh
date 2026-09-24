#!/bin/bash
# three 27B arms in sequence, each on its own temporary lmk (port 1236); the resident on 1235 is not touched
set -u
S="$1"; cd ~/src/lmk; export PYTHONPATH=$PWD/.engine/mlx-engine:$PWD
D=research/2026-09-23-intelligence-27b-vs-122b
start() { # name, model line
  H="$S/eval-$1"; mkdir -p "$H/logs"
  printf 'model: {%s}\nlisten: {port: 1236}\ncache: {dir: "%s/cache"}\nlog: {dir: "%s/logs"}\n' "$2" "$H" "$H" > "$H/config.yaml"
  LMK_HOME="$H" .venv/bin/python -P -m lmk serve > "$S/eval-$1.log" 2>&1 & echo $! > "$S/eval-$1.pid"
  until curl -s -m 2 http://127.0.0.1:1236/lmk/v1/status >/dev/null; do sleep 2; kill -0 "$(cat "$S/eval-$1.pid")" 2>/dev/null || { echo "$1 server died"; return 1; }; done
  echo "$(date +%H:%M) $1 server up"
}
stop() { kill "$(cat "$S/eval-$1.pid")" 2>/dev/null; wait "$(cat "$S/eval-$1.pid")" 2>/dev/null; sleep 3; echo "$(date +%H:%M) $1 server stopped"; }
run() { .venv/bin/python "$D/harness/run.py" "$1" http://127.0.0.1:1236 "$2" > "$S/eval-$1.progress" 2>&1; echo "$(date +%H:%M) $1 done: $(grep -c PASS "$S/eval-$1.progress") pass / $(grep -c ': fail' "$S/eval-$1.progress") fail"; }
# arm 1: the 27B-off server is already up (started for the smoke test)
run 27B-off qwen3.8-27b-4bit; stop 27B-off
start 27B-low  "name: qwen3.8-27b-4bit, reasoning_effort: low" && run 27B-low qwen3.8-27b-4bit; stop 27B-low
start 27B-xhigh "name: qwen3.8-27b-4bit" && run 27B-xhigh qwen3.8-27b-4bit; stop 27B-xhigh
echo "$(date +%H:%M) ALL 27B ARMS DONE"
