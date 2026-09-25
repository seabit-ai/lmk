#!/bin/sh
# exp01: cancel a prefill halfway, then re-send the same request (see README.md)
cd "$(dirname "$0")" && python3 run.py 2>&1 | tee result.log
