#!/bin/bash
# usage: ./run.sh <model dir>   — from the lmk checkout (.venv + .engine on the lmk branch)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE/../../.."
export PYTHONPATH="$PWD/.engine/mlx-engine:$PWD"
for BITS in 16 8; do
  H="$(mktemp -d)"; export LMK_HOME="$H"; mkdir -p "$H/logs"
  cat > "$H/config.yaml" <<YAML
model: {path: "$1", kv_cache_bits: $BITS}
listen: {port: 1236}
cache: {dir: "$H/cache"}
log: {dir: "$H/logs"}
YAML
  .venv/bin/python -P -m lmk serve > "$HERE/raw/serve-kv$BITS.log" 2>&1 &
  PID=$!
  until curl -s -m 2 http://127.0.0.1:1236/lmk/v1/status >/dev/null; do sleep 2; kill -0 $PID 2>/dev/null || { echo "serve died (kv$BITS)"; tail -5 "$HERE/raw/serve-kv$BITS.log"; break; }; done
  echo "== kv$BITS"; grep -o "Model context auto-fit.*" "$HERE/raw/serve-kv$BITS.log" | tail -1
  .venv/bin/python -m lmk bench --url http://127.0.0.1:1236 --seed 7 2>&1 | tee "$HERE/raw/bench-kv$BITS.txt"
  du -sk "$H/cache" | awk '{print "cache dir KB:", $1}' | tee "$HERE/raw/cache-size-kv$BITS.txt"
  kill $PID; wait $PID 2>/dev/null || true
  rm -rf "$H"
done
