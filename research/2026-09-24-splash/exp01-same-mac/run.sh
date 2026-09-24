#!/bin/bash
# One service at a time, same probes. Splash first, then lmk. Everything lands in raw/.
set -u
cd "$(dirname "$0")"
R="${1:-}"   # suffix for this run's files, e.g. -r2
PY=/Users/xinkai/src/lmk/.venv/bin/python
SPLASH_URL=http://127.0.0.1:8010; SPLASH_MODEL=incoai/Qwen3.8-27B-Splash
LMK_URL=http://127.0.0.1:1235;   LMK_MODEL=qwen3.8-27b-4bit
say() { echo "$(date +%H:%M:%S) $*"; }
wait_ready() {  # $1 = log, $2 = pid
  until grep -q "Ready" "$1" || ! kill -0 "$2" 2>/dev/null; do sleep 2; done
  grep -q "Ready" "$1" || { say "splash did not come up"; tail -5 "$1"; exit 1; }
}
start_splash() {  # $1 = log
  nohup splash serve --model $SPLASH_MODEL --port 8010 --default-reasoning-effort none --no-webui > "$1" 2>&1 &
  echo $! > raw/splash-pid.txt; wait_ready "$1" "$(cat raw/splash-pid.txt)"; say "splash ready ($(grep Ready "$1" | tail -1))"
}
stop_splash() {  # wait for the process to be gone: splash refuses to start while another one serves
  local pid; pid="$(cat raw/splash-pid.txt)"; kill "$pid" 2>/dev/null
  while kill -0 "$pid" 2>/dev/null; do sleep 1; done; sleep 1
}

say "stop the download-run splash and the resident lmk"
stop_splash; lmk down | tail -1
say "splash, clean start"; start_splash raw/splash-serve-clean$R.log
curl -s $SPLASH_URL/status > raw/splash-status$R.json; head -c 600 raw/splash-status$R.json; echo
say "probes on splash"; $PY -P probe.py $SPLASH_URL $SPLASH_MODEL raw/splash-probe$R.json 2>&1 | tee raw/splash-probe$R.txt
say "restart splash, replay the 32k prompt"; stop_splash; start_splash raw/splash-serve-restart$R.log
$PY -P probe.py $SPLASH_URL $SPLASH_MODEL raw/splash-replay$R.json --replay raw/splash-probe$R.json 2>&1 | tee raw/splash-replay$R.txt
stop_splash; say "splash stopped"
say "lmk up"; lmk up | grep -E "^(✓|✗)"
say "probes on lmk"; $PY -P probe.py $LMK_URL $LMK_MODEL raw/lmk-probe$R.json 2>&1 | tee raw/lmk-probe$R.txt
say "restart lmk, replay the 32k prompt"; lmk down | tail -1; lmk up | grep -E "^(✓|✗)"
$PY -P probe.py $LMK_URL $LMK_MODEL raw/lmk-replay$R.json --replay raw/lmk-probe$R.json 2>&1 | tee raw/lmk-replay$R.txt
say "DONE"
