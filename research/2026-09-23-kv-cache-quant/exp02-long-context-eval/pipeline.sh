#!/bin/bash
# three arms in sequence on a temporary lmk (port 1236). usage: pipeline.sh <scratch dir>
set -u
S="$1"; cd ~/src/lmk; export PYTHONPATH=$PWD/.engine/mlx-engine:$PWD
E=research/2026-09-23-kv-cache-quant/exp02-long-context-eval; H27=research/2026-09-23-intelligence-27b-vs-122b/harness
export EVAL_RAW_DIR=$PWD/$E/raw EVAL_PREFIX_FILE=$PWD/$E/prefix-60k.txt EVAL_RUNS=2 EVAL_LIMIT=20
start() { H="$S/kvq-$1"; mkdir -p "$H/logs"
  printf 'model: {name: qwen3.8-27b-4bit, reasoning_effort: low, kv_cache_bits: %s}\nlisten: {port: 1236}\ncache: {dir: "%s/cache"}\nlog: {dir: "%s/logs"}\n' "$2" "$H" "$H" > "$H/config.yaml"
  LMK_HOME="$H" .venv/bin/python -P -m lmk serve > "$S/kvq-$1.log" 2>&1 & echo $! > "$S/kvq-$1.pid"
  until curl -s -m 2 http://127.0.0.1:1236/lmk/v1/status >/dev/null; do sleep 2; kill -0 "$(cat "$S/kvq-$1.pid")" 2>/dev/null || { echo "$1 server died"; return 1; }; done
  echo "$(date +%H:%M) $1 server up"
  # warm the 60k prefix once so the arm's requests hit the cache instead of two cold prefills racing
  .venv/bin/python - <<PY
import json, os, urllib.request
prefix = open(os.environ["EVAL_PREFIX_FILE"]).read()
body = {"model": "qwen3.8-27b-4bit", "max_tokens": 8, "messages": [{"role": "system", "content": "Reference material — the files of the project you are working in (for context; the task follows):\n\n" + prefix}, {"role": "user", "content": "Say ok."}]}
req = urllib.request.Request("http://127.0.0.1:1236/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=3600).read()); print("warm-up prompt tokens", d["usage"]["prompt_tokens"])
PY
}
stop() { kill "$(cat "$S/kvq-$1.pid")" 2>/dev/null; wait "$(cat "$S/kvq-$1.pid")" 2>/dev/null; sleep 3; echo "$(date +%H:%M) $1 server stopped"; }
run() { .venv/bin/python "$H27/run.py" "$1" http://127.0.0.1:1236 qwen3.8-27b-4bit code instruct tools > "$S/kvq-$1.progress" 2>&1
        echo "$(date +%H:%M) $1 done: $(grep -c PASS "$S/kvq-$1.progress") pass / $(grep -c ': fail' "$S/kvq-$1.progress") fail"; }
lmk down >/dev/null
for ARM in kv16 kv8 kv4; do start $ARM ${ARM#kv} && run $ARM; stop $ARM; done
lmk up >/dev/null; echo "$(date +%H:%M) ALL ARMS DONE, resident back"
EVAL_RAW_DIR=$PWD/$E/raw .venv/bin/python "$H27/run.py" grade | tee "$E/grade.md"
