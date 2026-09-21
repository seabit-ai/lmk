#!/usr/bin/env python3
"""usage: run.py <label> <n_parallel> [--late-cold]   (see README.md)"""
import json
import sys
import threading
import time
import urllib.request
import uuid

URL = "http://127.0.0.1:1235"
label, n = sys.argv[1], int(sys.argv[2])
late_cold = "--late-cold" in sys.argv
model = json.load(urllib.request.urlopen(URL + "/v1/models"))["data"][0]["id"]


def chat(name, messages, max_tokens, out, delay=0.0):
    time.sleep(delay)
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": max_tokens, "messages": messages}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Lmk-Purpose": f"exp01-{label}",
                                          "X-Lmk-Ref-Id": name})
    t0 = time.monotonic()
    ttft, usage, stamps = None, None, []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                    now = time.monotonic() - t0
                    ttft = now if ttft is None else ttft
                    stamps.append(round(now + delay, 2))  # on the experiment's clock
    total = time.monotonic() - t0
    out[name] = {"ttft_s": round(ttft, 2), "total_s": round(total, 2), "prompt_tokens": usage["prompt_tokens"],
                 "completion_tokens": usage["completion_tokens"],
                 "tok_per_s": round(usage["completion_tokens"] / max(total - ttft, 1e-6), 1), "stamps": stamps}


def counting(i):
    return [{"role": "user", "content": f"[{uuid.uuid4().hex[:6]}] Count upward from {i * 1000 + 1}, one number per line. "
                                        "Do not stop, do not explain."}]


results, threads = {}, []
for i in range(n):
    threads.append(threading.Thread(target=chat, args=(f"gen-{i}", counting(i), 300, results)))
if late_cold:
    nonce = uuid.uuid4().hex
    cold = " ".join(f"Rule {nonce[:4]}-{k}: never mention the number {k * 7919}." for k in range(620))
    threads.append(threading.Thread(target=chat, args=("cold-late", [{"role": "system", "content": cold},
                                    {"role": "user", "content": "Say hi."}], 16, results, 2.0)))
for t in threads:
    t.start()
for t in threads:
    t.join()

for name, r in sorted(results.items()):
    stamps = r.pop("stamps")
    if late_cold and name == "gen-0":
        b = results.get("cold-late") or {}
        lo, hi = 2.0, 2.0 + (b.get("ttft_s") or 0)
        during = [s for s in stamps if lo <= s <= hi]
        r["chunks_while_other_read_its_prompt"] = len(during)
        r["that_window_s"] = round(hi - lo, 1)
    print(json.dumps({"label": label, "request": name, **r}))
