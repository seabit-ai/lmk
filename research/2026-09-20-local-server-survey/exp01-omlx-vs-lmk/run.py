#!/usr/bin/env python3
"""One streamed chat-completions call; prints a one-line JSON measurement.

usage: run.py <base_url> <model> <payload.json> <label> [--user TEXT] [--key KEY]
Raw SSE is saved to $EXP_RAW_DIR/<label>.sse. Payloads and raw output come from real
sessions (private), so they live outside this repo: ~/src/nova/2026/2026.0920.M3uOmlxVsLmk/.
"""
import json
import os
import pathlib
import sys
import time
import urllib.request

raw_dir = pathlib.Path(os.environ["EXP_RAW_DIR"])
base, model, payload_path, label = sys.argv[1:5]
opts = sys.argv[5:]
user = opts[opts.index("--user") + 1] if "--user" in opts else None
key = opts[opts.index("--key") + 1] if "--key" in opts else None

body = json.loads(pathlib.Path(payload_path).read_text())
body["model"] = model
if user is not None:
    body["messages"][-1] = {"role": "user", "content": user}
headers = {"Content-Type": "application/json"}
if key:
    headers["Authorization"] = f"Bearer {key}"
req = urllib.request.Request(
    base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers
)

t0 = time.monotonic()
ttft = None
usage = None
finish = None
text, reasoning, tool_calls = [], [], {}
raw = open(raw_dir / f"{label}.sse", "wb")
with urllib.request.urlopen(req, timeout=1800) as resp:
    for line in resp:
        raw.write(line)
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            d = ch.get("delta") or {}
            r = d.get("reasoning_content") or d.get("reasoning")
            has = bool(d.get("content") or r or d.get("tool_calls"))
            if has and ttft is None:
                ttft = time.monotonic() - t0
            if d.get("content"):
                text.append(d["content"])
            if r:
                reasoning.append(r)
            for tc in d.get("tool_calls") or []:
                slot = tool_calls.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["args"] += fn.get("arguments") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
raw.close()
total = time.monotonic() - t0

calls = []
for slot in tool_calls.values():
    try:
        json.loads(slot["args"])
        ok = True
    except ValueError:
        ok = False
    calls.append({"name": slot["name"], "args_json_ok": ok, "args": slot["args"][:120]})
details = (usage or {}).get("prompt_tokens_details") or {}
print(json.dumps({
    "label": label,
    "ttft_s": round(ttft, 2) if ttft is not None else None,
    "total_s": round(total, 2),
    "prompt_tokens": (usage or {}).get("prompt_tokens"),
    "cached_tokens": details.get("cached_tokens"),
    "completion_tokens": (usage or {}).get("completion_tokens"),
    "finish": finish,
    "reasoning_chars": len("".join(reasoning)),
    "text": "".join(text)[:160],
    "tool_calls": calls,
}, ensure_ascii=False))
