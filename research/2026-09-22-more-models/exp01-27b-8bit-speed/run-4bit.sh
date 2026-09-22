#!/bin/sh
# usage: ./run.sh <model dir>   — run from the lmk checkout (needs .venv + .engine)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/../../.."
H="$(mktemp -d)"; export LMK_HOME="$H"
mkdir -p "$H/logs"
cat > "$H/config.yaml" <<YAML
model: {path: "$1", id: probe}
listen: {port: 1236}
cache: {dir: "$H/cache"}
log: {dir: "$H/logs"}
YAML
export PYTHONPATH="$PWD/.engine/mlx-engine:$PWD"
t0=$(date +%s)
.venv/bin/python -P -m lmk serve > "$HERE/raw-4bit/serve.log" 2>&1 &
PID=$!
until curl -s -m 2 http://127.0.0.1:1236/lmk/v1/status >/dev/null; do sleep 2; kill -0 $PID || { echo "serve died"; exit 1; }; done
echo "loaded in $(( $(date +%s) - t0 ))s"
LONG=$(python3 -c "print(('The quick brown fox jumps over the lazy dog. ' * 400))")
curl -s -m 600 http://127.0.0.1:1236/v1/chat/completions -H 'content-type: application/json' \
  -d "{\"model\":\"probe\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG Reply with the single word: done\"}],\"temperature\":0,\"max_tokens\":32}" > "$HERE/raw-4bit/cold.json"
curl -s -m 600 http://127.0.0.1:1236/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"probe","messages":[{"role":"user","content":"Write a 250-word story about a lighthouse keeper."}],"temperature":0,"max_tokens":400}' > "$HERE/raw-4bit/decode.json"
kill $PID; wait $PID 2>/dev/null || true
cp "$H/logs/lmk.jsonl" "$HERE/raw-4bit/lmk.jsonl"
python3 - "$HERE/raw-4bit/lmk.jsonl" <<'PY'
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("event") == "LmkChatDone":
        u = r["promptTokens"] - r["cachedTokens"]; tt = r["ttftMs"] - (r.get("restoreMs") or 0); c = r["completionTokens"]
        dec = r["totalMs"] - r["ttftMs"]
        print(f"prompt {r['promptTokens']} uncached {u}: ttft {r['ttftMs']} ms -> prefill {u/tt*1000:.0f} tok/s | "
              f"completion {c} in {dec} ms -> decode {c/dec*1000 if dec else 0:.1f} tok/s")
PY
rm -rf "$H"
