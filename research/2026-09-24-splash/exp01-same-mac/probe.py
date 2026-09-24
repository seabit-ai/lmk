"""Same probes against any OpenAI-compatible server: lmk's bench four, a ~32k-token cold+hit pair, and 4 concurrent short prompts.

usage: probe.py <url> <model id> <out.json> [--skip-long] [--skip-concurrent]
Timing is client-side (HTTP included), temperature 0, one request at a time except the concurrency probe.
`cached_tokens` is read when the server reports it (lmk does; others may not): the hit is judged by first-token time either way.
"""
import json, sys, threading, time, urllib.error, urllib.request
sys.path.insert(0, "/Users/xinkai/src/lmk")
from lmk import bench

url, model_id, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
LONG_REPEATS = 3200          # ~32k tokens of the bench sentence (4,060 tokens per 400 repeats on Qwen3.8)
CONCURRENT = 4


def stream(body):
    body = dict(body, stream_options={"include_usage": True})
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); chunks = []
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    chunks.append((int((time.perf_counter() - t0) * 1000), json.loads(line[6:])))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
    for _, c in chunks:
        if c.get("error"):
            raise RuntimeError(f"error chunk: {c['error']}")
    # the first token is the first NON-EMPTY piece of text: Splash opens its stream with a role chunk whose
    # content is "" (lmk's bench looked for the key, not the text, and read a 4k prefill as 0.01 s)
    for chunk in chunks:
        delta = (chunk[1].get("choices") or [{}])[0].get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content"):
            break
    else:
        raise RuntimeError("no text in the stream")
    chunks = [(ms, c) for ms, c in chunks if not (c.get("choices") and not ((c["choices"][0].get("delta") or {}).get("content")
                                                                           or (c["choices"][0].get("delta") or {}).get("reasoning_content"))
                                                  and not c.get("usage"))]
    if not any(c.get("usage") for _, c in chunks):
        raise RuntimeError("no usage chunk: cannot count tokens")
    return chunks


def probe(messages, max_tokens):
    return bench._probe(stream, model_id, messages, max_tokens)


def as_dict(p):
    return {k: getattr(p, k) for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "first_token_ms", "total_ms", "restore_ms", "draft_accepted", "draft_drafted")}


if "--replay" in sys.argv:
    prev = json.load(open(sys.argv[sys.argv.index("--replay") + 1]))
    nonce = bench.nonce_for(prev["bench"]["seed"] + 1)
    long_prompt = [{"role": "user", "content": f"Benchmark run {nonce}.\n{bench.TEMPLATE_SENTENCE * LONG_REPEATS}Reply with the single word: done"}]
    hit = probe(long_prompt, 16)
    json.dump({"after_restart_long_hit": as_dict(hit)}, open(out, "w"), indent=1)
    print(f"after restart    {hit.first_token_ms/1000:7.2f} s first token on the 32k prompt ({hit.cached_tokens} cached reported; cold was {prev['long']['cold']['first_token_ms']/1000:.2f} s)")
    sys.exit(0)

results = {"url": url, "model": model_id, "date": time.strftime("%Y-%m-%d %H:%M")}
r = bench.run_bench(stream, model_id, say=print)
results["bench"] = {"seed": r.seed, "warmup": as_dict(r.warmup), "cold": as_dict(r.cold), "hit": as_dict(r.hit), "decode": as_dict(r.decode), "decode_code": as_dict(r.decode_code)}
print(f"cold prefill     {r.cold_prefill_tok_s:7.0f} tok/s  ({r.cold.prompt_tokens} prompt, {r.cold.cached_tokens} cached, first token {r.cold.first_token_ms/1000:.2f} s)")
print(f"hit first token  {r.hit_first_token_s:7.2f} s     ({r.hit.cached_tokens} cached reported)")
print(f"decode prose     {r.decode_tok_s:7.1f} tok/s  code {r.decode_code_tok_s:.1f} tok/s  (accepted {r.decode_code.acceptance})", flush=True)

if "--skip-long" not in sys.argv:
    nonce = bench.nonce_for(r.seed + 1)
    long_prompt = [{"role": "user", "content": f"Benchmark run {nonce}.\n{bench.TEMPLATE_SENTENCE * LONG_REPEATS}Reply with the single word: done"}]
    cold = probe(long_prompt, 16); hit = probe(long_prompt, 16)
    results["long"] = {"cold": as_dict(cold), "hit": as_dict(hit), "prompt": long_prompt[0]["content"][:80]}
    print(f"long cold        {cold.first_token_ms/1000:7.2f} s first token ({cold.prompt_tokens} prompt tokens)  ->  hit {hit.first_token_ms/1000:.2f} s ({hit.cached_tokens} cached reported)", flush=True)

def concurrent(n):
    prompts = ["Write a 250-word story about a lighthouse keeper.", "Write a Python class implementing an LRU cache with get and put.",
               "Explain how a hash table handles collisions, in 250 words.", "Write a bash script that renames files to lower case, with comments."]
    got, errors = {}, {}
    def worker(i):
        try:
            got[i] = probe([{"role": "user", "content": prompts[i]}], 400)
        except Exception as e:  # noqa: BLE001 - a server failure is a result here, not a crash of the probe
            errors[i] = str(e)[:200]
    t0 = time.perf_counter(); ths = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    [t.start() for t in ths]; [t.join() for t in ths]; wall = time.perf_counter() - t0
    total = sum(p.completion_tokens for p in got.values())
    results[f"concurrent_{n}"] = {"n": n, "wall_s": wall, "total_tokens": total, "aggregate_tok_s": total / wall,
                                  "each": [as_dict(got[i]) for i in sorted(got)], "errors": errors}
    print(f"concurrent {n}     {total/wall:7.1f} tok/s aggregate ({total} tokens in {wall:.1f} s; first tokens "
          f"{[round(got[i].first_token_ms/1000,2) for i in sorted(got)]}; errors {len(errors)} {list(errors.values())[:1]})", flush=True)


if "--skip-concurrent" not in sys.argv:
    concurrent(2)
    concurrent(4)

json.dump(results, open(out, "w"), indent=1)
print("wrote", out)


# --replay <results.json>: send that run's long prompt once more (after a server restart) and report the first-token time.
