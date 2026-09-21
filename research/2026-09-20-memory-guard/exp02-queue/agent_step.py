#!/usr/bin/env python3
"""A keeps writing; 2s later an agent's next step arrives: a 27k-token conversation lmk has cached.
It must NOT wait for A. Prints both requests' timings. Payload: the private one in nova (see notes)."""
import json, pathlib, sys, threading, time, urllib.request
URL = "http://127.0.0.1:1235"
model = json.load(urllib.request.urlopen(URL + "/v1/models"))["data"][0]["id"]
out = {}
def chat(name, body, delay):
    time.sleep(delay)
    body = {**body, "model": model, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Lmk-Purpose": "exp02-agent-step", "X-Lmk-Ref-Id": name})
    t0, ttft, usage = time.monotonic(), None, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                continue
            c = json.loads(line[6:]); usage = c.get("usage") or usage
            if ttft is None and any((ch.get("delta") or {}).get(k) for ch in c.get("choices") or [] for k in ("content", "reasoning_content", "tool_calls")):
                ttft = time.monotonic() - t0
    out[name] = {"ttft_s": round(ttft, 2), "total_s": round(time.monotonic() - t0, 2), "prompt_tokens": usage["prompt_tokens"],
                 "cached_tokens": usage["prompt_tokens_details"]["cached_tokens"], "completion_tokens": usage["completion_tokens"]}
writing = {"max_tokens": 400, "messages": [{"role": "user", "content": "Count upward from 1, one number per line. Do not stop."}]}
step = json.loads(pathlib.Path(sys.argv[1]).read_text()); step["max_tokens"] = 40
ts = [threading.Thread(target=chat, args=("writing", writing, 0)), threading.Thread(target=chat, args=("agent-step", step, 2.0))]
[t.start() for t in ts]; [t.join() for t in ts]
for k, v in out.items():
    print(json.dumps({"request": k, **v}))
