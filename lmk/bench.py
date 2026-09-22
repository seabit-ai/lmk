"""`lmk bench`: the numbers a user feels, measured from outside against the running service.

A warm-up first, then three probes, temperature 0, one at a time:
  warm-up       — a tiny request. After hours idle, or after other models were loaded, the first
                  touch of the weights pages them back in (measured: 37 s instead of 12.8 s for the
                  cold probe, research/2026-09-22-more-models BNC-001). Reported on its own line.
  cold prefill  — a fixed ~4k-token text behind a random number, so no cached prefix can match
                  (the cache restores up to the longest common prefix; a fresh first token makes that zero)
  cache hit     — the same request again: the whole prompt comes back from disk
  decode        — a short prompt, 400 tokens out
Timing is client-side (HTTP included): what an agent on this Mac would see.
The result is one Markdown row for docs/benchmarks.md."""
import json
import random
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from lmk.clock import get_current_clock

TEMPLATE_SENTENCE = "The quick brown fox jumps over the lazy dog. "
TEMPLATE_REPEATS = 400          # ~4,060 tokens on Qwen3.8 (research/2026-09-22-more-models exp01)
DECODE_TOKENS = 400
DECODE_PROMPT = "Write a 250-word story about a lighthouse keeper."


def probe_prompt(nonce: str) -> str:
    return f"Benchmark run {nonce}.\n{TEMPLATE_SENTENCE * TEMPLATE_REPEATS}Reply with the single word: done"


def new_nonce(rng: Optional[random.Random] = None) -> str:
    return "".join(str((rng or random).randint(0, 9)) for _ in range(10))   # fixed length: fixed token count


@dataclass
class Probe:
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    first_token_ms: int
    total_ms: int


@dataclass
class BenchResult:
    warmup: Probe
    cold: Probe
    hit: Probe
    decode: Probe

    @property
    def cold_prefill_tok_s(self) -> float:
        return (self.cold.prompt_tokens - self.cold.cached_tokens) / self.cold.first_token_ms * 1000

    @property
    def hit_first_token_s(self) -> float:
        return self.hit.first_token_ms / 1000

    @property
    def decode_tok_s(self) -> float:
        return self.decode.completion_tokens / max(1, self.decode.total_ms - self.decode.first_token_ms) * 1000


StreamFn = Callable[[dict], "list[tuple[int, dict]]"]   # body -> [(ms since start, chunk)]


def stream_via_http(url: str, timeout_s: float = 900) -> StreamFn:
    clock = get_current_clock()

    def run(body: dict):
        req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "X-Lmk-Purpose": "bench"})
        started = clock.mono_ms()
        out = []
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            for line in r:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    out.append((clock.mono_ms() - started, json.loads(line[6:])))
        return out
    return run


def _probe(stream: StreamFn, model_id: str, messages: list, max_tokens: int) -> Probe:
    chunks = stream({"model": model_id, "stream": True, "temperature": 0, "max_tokens": max_tokens, "messages": messages})
    first = next(ms for ms, c in chunks
                 if c.get("choices") and any(k in c["choices"][0]["delta"] for k in ("content", "reasoning_content")))
    usage = next(c["usage"] for _, c in chunks if c.get("usage"))
    return Probe(prompt_tokens=usage["prompt_tokens"],
                 cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                 completion_tokens=usage["completion_tokens"], first_token_ms=first, total_ms=chunks[-1][0])


def run_bench(stream: StreamFn, model_id: str, nonce: Optional[str] = None,
              say: Callable[[str], None] = lambda _: None) -> BenchResult:
    say("· warm-up: a tiny request, so paging the weights back in is not charged to the numbers")
    warmup = _probe(stream, model_id, [{"role": "user", "content": "Reply with the single word: ready"}], 8)
    prompt = [{"role": "user", "content": probe_prompt(nonce or new_nonce())}]
    say("· cold prefill: a ~4k-token prompt no one has sent before")
    cold = _probe(stream, model_id, prompt, 32)
    say("· cache hit: the same prompt again")
    hit = _probe(stream, model_id, prompt, 32)
    say(f"· decode: {DECODE_TOKENS} tokens out")
    decode = _probe(stream, model_id, [{"role": "user", "content": DECODE_PROMPT}], DECODE_TOKENS)
    return BenchResult(warmup=warmup, cold=cold, hit=hit, decode=decode)


def machine() -> dict:
    def sysctl(key: str) -> str:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "?"
    mem = sysctl("hw.memsize")
    return {"chip": sysctl("machdep.cpu.brand_string") or "?",
            "memory_gb": round(int(mem) / 1024**3) if mem.isdigit() else None}


ROW_HEADER = ("| chip | memory | model | context | cold prefill | cache-hit first token | decode | lmk | engine | date |\n"
              "|---|---|---|---|---|---|---|---|---|---|")


def markdown_row(r: BenchResult, m: dict, status: dict, date: str) -> str:
    model = status["model"]
    mem = f"{m['memory_gb']} GB" if m.get("memory_gb") else "?"
    return (f"| {m['chip']} | {mem} | {model['id']} | {model['context_length']:,} | "
            f"{r.cold_prefill_tok_s:.0f} tok/s ({r.cold.prompt_tokens - r.cold.cached_tokens:,} tokens) | "
            f"{r.hit_first_token_s:.2f} s ({r.hit.cached_tokens:,} cached) | {r.decode_tok_s:.1f} tok/s | "
            f"{status.get('build', '?')} | {str(status.get('engine', '?'))[:7]} | {date} |")


def human_block(r: BenchResult) -> str:
    return "\n".join([
        f"  warm-up           first token after {r.warmup.first_token_ms / 1000:.1f} s   (not in the numbers below; "
        "more than a few seconds = the weights were paged back in)",
        f"  cold prefill      {r.cold_prefill_tok_s:.0f} tokens/s   "
        f"({r.cold.prompt_tokens - r.cold.cached_tokens:,} uncached tokens, first token after {r.cold.first_token_ms / 1000:.1f} s)",
        f"  cache hit         first token after {r.hit_first_token_s:.2f} s   ({r.hit.cached_tokens:,} of {r.hit.prompt_tokens:,} tokens from the cache)",
        f"  decode            {r.decode_tok_s:.1f} tokens/s   ({r.decode.completion_tokens} tokens)",
    ])
