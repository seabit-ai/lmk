#!/usr/bin/env python3
import json, pathlib, sys, threading, time, urllib.request
URL = "http://127.0.0.1:1235"
label, n, payload = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
model = json.load(urllib.request.urlopen(URL + "/v1/models"))["data"][0]["id"]
body = {**json.loads(payload.read_text()), "model": model, "stream": True, "max_tokens": 200,
        "stream_options": {"include_usage": True}}
out = {}
def chat(name):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Lmk-Purpose": f"exp03-{label}", "X-Lmk-Ref-Id": name})
    t0, ttft, usage = time.monotonic(), None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            c = json.loads(line[6:]); usage = c.get("usage") or usage
            if ttft is None and any((ch.get("delta") or {}).get(k) for ch in c.get("choices") or [] for k in ("content", "reasoning_content", "tool_calls")):
                ttft = time.monotonic() - t0
    total = time.monotonic() - t0
    out[name] = {"ttft_s": round(ttft, 2), "cached": usage["prompt_tokens_details"]["cached_tokens"],
                 "completion_tokens": usage["completion_tokens"], "tok_per_s": round(usage["completion_tokens"] / (total - ttft), 1)}
ts = [threading.Thread(target=chat, args=(f"long-{i}",)) for i in range(n)]
[t.start() for t in ts]; [t.join() for t in ts]
for k in sorted(out):
    print(json.dumps({"label": label, "request": k, **out[k]}))
