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


def new_seed() -> int:
    return random.randint(1, 999_999)


def nonce_for(seed: int) -> str:
    """Ten digits from the seed: fixed length, so the token count does not move. The same seed
    gives the same prompt — run bench again with it after a restart and the cold probe hits the cache."""
    rng = random.Random(seed)
    return "".join(str(rng.randint(0, 9)) for _ in range(10))


@dataclass
class Probe:
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    first_token_ms: int
    total_ms: int
    restore_ms: Optional[int] = None   # server-side: the cached part coming back from disk (lmk's usage chunk)


@dataclass
class BenchResult:
    seed: int
    warmup: Probe
    cold: Probe
    hit: Probe
    decode: Probe

    @property
    def cold_was_cold(self) -> bool:
        """False when the seed was reused and the 'cold' prompt came back from the cache."""
        return self.cold.cached_tokens == 0

    @property
    def cold_prefill_tok_s(self) -> float:
        return (self.cold.prompt_tokens - self.cold.cached_tokens) / self.cold.first_token_ms * 1000

    @property
    def hit_first_token_s(self) -> float:
        return self.hit.first_token_ms / 1000

    @property
    def hit_tok_s(self) -> Optional[float]:
        """How fast the cached part comes back from disk, from the server's own restore timing.
        None when the server did not report it (an older lmk)."""
        if self.hit.restore_ms is None or self.hit.cached_tokens == 0:
            return None
        return self.hit.cached_tokens / max(1, self.hit.restore_ms) * 1000

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
    usage_chunk = next(c for _, c in chunks if c.get("usage"))
    usage = usage_chunk["usage"]
    return Probe(prompt_tokens=usage["prompt_tokens"],
                 cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                 completion_tokens=usage["completion_tokens"], first_token_ms=first, total_ms=chunks[-1][0],
                 restore_ms=(usage_chunk.get("lmk") or {}).get("restore_ms"))


def run_bench(stream: StreamFn, model_id: str, seed: Optional[int] = None,
              say: Callable[[str], None] = lambda _: None) -> BenchResult:
    seed = new_seed() if seed is None else seed
    warmup = _probe(stream, model_id, [{"role": "user", "content": "Reply with the single word: ready"}], 8)
    prompt = [{"role": "user", "content": probe_prompt(nonce_for(seed))}]
    cold = _probe(stream, model_id, prompt, 32)
    hit = _probe(stream, model_id, prompt, 32)
    decode = _probe(stream, model_id, [{"role": "user", "content": DECODE_PROMPT}], DECODE_TOKENS)
    return BenchResult(seed=seed, warmup=warmup, cold=cold, hit=hit, decode=decode)


def machine() -> dict:
    def sysctl(key: str) -> str:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "?"
    mem = sysctl("hw.memsize")
    return {"chip": sysctl("machdep.cpu.brand_string") or "?",
            "memory_gb": round(int(mem) / 1024**3) if mem.isdigit() else None}


ROW_HEADER = ("| chip | memory | model | context | prefill | cached prefill | decode | lmk | engine | date |\n"
              "|---|---|---|---|---|---|---|---|---|---|")


def markdown_row(r: BenchResult, m: dict, status: dict, date: str) -> str:
    model = status["model"]
    mem = f"{m['memory_gb']} GB" if m.get("memory_gb") else "?"
    cold = (f"{r.cold_prefill_tok_s:.0f} tok/s ({r.cold.prompt_tokens - r.cold.cached_tokens:,} tokens)" if r.cold_was_cold
            else f"— (seed reused: {r.cold.cached_tokens:,} cached)")
    return (f"| {m['chip']} | {mem} | {model['id']} | {model['context_length']:,} | {cold} | "
            f"{_k(r.hit_tok_s)} tok/s ({r.hit.cached_tokens:,} cached; first token {r.hit_first_token_s:.2f} s) | {r.decode_tok_s:.1f} tok/s | "
            f"{status.get('build', '?')} | {str(status.get('engine', '?'))[:7]} | {date} |")


def _k(tok_s: Optional[float]) -> str:
    return "?" if tok_s is None else f"{tok_s / 1000:.0f}k"


def human_block(r: BenchResult) -> str:
    cold = (f"{r.cold_prefill_tok_s:>7.0f} tokens/s" if r.cold_was_cold
            else f"      —           (seed {r.seed} reused and the cache still had it)")
    hit = f"{_k(r.hit_tok_s):>7} tokens/s" if r.hit_tok_s is not None else "       ? tokens/s   (this lmk does not report restore time)"
    lines = [f"  prefill          {cold}",
             f"  cached prefill   {hit}   ({r.hit.cached_tokens:,} of {r.hit.prompt_tokens:,} tokens from disk; "
             f"first token after {r.hit_first_token_s:.2f} s, the rest is the last partial block computed)",
             f"  decode           {r.decode_tok_s:>7.1f} tokens/s"]
    if r.warmup.first_token_ms > 3000:
        lines.append(f"  (the warm-up request took {r.warmup.first_token_ms / 1000:.0f} s: the weights had to be paged back in; not counted)")
    return "\n".join(lines)
