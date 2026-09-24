#!/bin/bash
# Extra Splash tests after exp01 run 2: is the draft doing anything, and what changes decode speed. One service at a time.
set -u
cd "$(dirname "$0")"
PY=/Users/xinkai/src/lmk/.venv/bin/python
URL=http://127.0.0.1:8010; MODEL=incoai/Qwen3.8-27B-Splash
say() { echo "$(date +%H:%M:%S) $*"; }
start_splash() {  # $1 = log, rest = extra flags
  local log=$1; shift
  nohup splash serve --model $MODEL --port 8010 --no-webui "$@" > "$log" 2>&1 &
  echo $! > raw/splash-pid.txt
  until grep -q "Ready" "$log" || ! kill -0 "$(cat raw/splash-pid.txt)" 2>/dev/null; do sleep 2; done
  grep -q "Ready" "$log" || { say "splash did not come up: $*"; tail -3 "$log"; return 1; }
  say "splash ready with flags: $*"
}
stop_splash() { local pid; pid="$(cat raw/splash-pid.txt)"; kill "$pid" 2>/dev/null; while kill -0 "$pid" 2>/dev/null; do sleep 1; done; sleep 1; }
R() { $PY -P req.py $URL $MODEL "$@"; }

lmk down | tail -1
say "T1 baseline flags (effort none)"; start_splash raw/extra-T1.log --default-reasoning-effort none || exit 1
R T1-warm --prose --max-tokens 8
R T1-code-temp0 --code --status
R T1-prose-temp0 --prose
R T1-code-default-sampling --code --no-temp --status
R T1-code-1000 --long --max-tokens 1000 --status
R T1-code-nostream --code --no-stream
say "T1 two at once"; ( R T1-pair-a --code & R T1-pair-b --prose & wait )
$PY - <<'PY'
import json, urllib.request
s = json.load(urllib.request.urlopen("http://127.0.0.1:8010/status")); t = s.get("transport", {})
print("   transport:", {k: t.get(k) for k in ("ready", "recovering", "restarts")}, "last_crash_trace:", str(t.get("last_crash_trace"))[:600])
json.dump(s, open("raw/splash-status-T1-after-pair.json", "w"), indent=1)
PY
stop_splash
say "T2 --kv-format bf16"; start_splash raw/extra-T2.log --default-reasoning-effort none --kv-format bf16 && { R T2-warm --prose --max-tokens 8; R T2-code-temp0 --code --status; R T2-prose-temp0 --prose; stop_splash; }
say "T3 --max-context 32K"; start_splash raw/extra-T3.log --default-reasoning-effort none --max-context 32K && { R T3-warm --prose --max-tokens 8; R T3-code-temp0 --code --status; stop_splash; }
say "T4 template default effort (thinking on)"; start_splash raw/extra-T4.log && { R T4-warm --prose --max-tokens 8; R T4-code-temp0 --code --max-tokens 1200 --status; stop_splash; }
say "splash log lines:"; grep -h "Done\|Error\|error" raw/extra-T*.log | cut -c1-160
lmk up | grep -E "^(✓|✗)"
say "DONE"
