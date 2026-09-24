"""One request against an OpenAI-compatible server; prints client-side timing and, for Splash, its /status draft metrics.
usage: req.py <url> <model> <label> [--prose|--code|--long] [--no-temp] [--no-stream] [--max-tokens N] [--status]
"""
import json, sys, time, urllib.request
url, model, label = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
a = sys.argv[4:]
prompt = {"--prose": "Write a 250-word story about a lighthouse keeper.",
          "--code": "Write a Python class implementing an LRU cache with get and put, with type hints and docstrings.",
          "--long": "Write a complete Python module implementing a small key-value store with a CLI: classes, argument parsing, tests. Be thorough."}
kind = next((k for k in prompt if k in a), "--code")
max_tokens = int(a[a.index("--max-tokens") + 1]) if "--max-tokens" in a else 400
body = {"model": model, "messages": [{"role": "user", "content": prompt[kind]}], "max_tokens": max_tokens, "stream": "--no-stream" not in a}
if "--no-temp" not in a:
    body["temperature"] = 0
if body["stream"]:
    body["stream_options"] = {"include_usage": True}
req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
t0 = time.perf_counter(); first = None; n_text = 0; usage = None; text = ""
with urllib.request.urlopen(req, timeout=1800) as r:
    if body["stream"]:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("error"):
                print(label, "ERROR", c["error"]); sys.exit(1)
            d = (c.get("choices") or [{}])[0].get("delta") or {}
            if d.get("content") or d.get("reasoning_content"):
                if first is None:
                    first = time.perf_counter()
                text += d.get("content") or ""
                n_text += 1
            if c.get("usage"):
                usage = c["usage"]
    else:
        c = json.loads(r.read()); usage = c.get("usage"); first = time.perf_counter(); text = c["choices"][0]["message"].get("content") or ""
end = time.perf_counter()
out = usage.get("completion_tokens") if usage else None
rate = out / (end - first) if (out and first and end > first) else None
print(f"{label:34} ttft {((first or end) - t0):5.2f} s  out {out} tokens  {rate and round(rate, 1)} tok/s (client, after first token)  text[:60]={text[:60]!r}")
if "--status" in a:
    s = json.load(urllib.request.urlopen(f"{url}/status"))
    m = s.get("metrics", {}); w = s.get("warmup", {})
    print(f"   /status metrics: decode {m.get('decode_tokens_per_second')} tok/s · drafted {m.get('drafted_tokens')} accepted {m.get('accepted_draft_tokens')} rate {m.get('draft_acceptance_rate')} · itl {m.get('itl_ms')} ms · metal_failures {m.get('metal_failures')}")
    print(f"   /status warmup: decode_b1 {w.get('decode_b1')} b2 {w.get('decode_b2')} b4 {w.get('decode_b4')} draft_verify_commit {w.get('draft_verify_commit')} detail {str(w.get('detail'))[:200]}")
    json.dump(s, open(f"raw/splash-status-{label}.json", "w"), indent=1)
