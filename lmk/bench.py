"""`lmk bench`: the numbers a user feels, measured from outside against the running service.

A warm-up first, then three probes, temperature 0, one at a time:
  warm-up       — a tiny request. After hours idle, or after other models were loaded, the first
                  touch of the weights pages them back in (measured: 37 s instead of 12.8 s for the
                  cold probe, research/2026-09-22-more-models BNC-001). Reported on its own line.
  cold prefill  — a fixed ~4k-token text behind a random number, so no cached prefix can match
                  (the cache restores up to the longest common prefix; a fresh first token makes that zero)
  cache hit     — the same request again: the whole prompt comes back from disk
  decode        — a short prompt, 400 tokens out: prose (a story), then code (a class). Both, because
                  speculative decoding gains most on code (exp05: 1.49x code, 1.14x prose) — with one prose
                  number a user who turned it on sees almost nothing (owner, 2026-09-24)
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
DECODE_CODE_PROMPT = "Write a Python class implementing an LRU cache with get and put, with type hints and docstrings."


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
    draft_accepted: Optional[int] = None  # speculative decoding: drafted tokens the model agreed with ...
    draft_drafted: Optional[int] = None   # ... out of how many it drafted; None when no draft model is loaded

    @property
    def acceptance(self) -> Optional[float]:
        if not self.draft_drafted:
            return None
        return (self.draft_accepted or 0) / self.draft_drafted


@dataclass
class BenchResult:
    seed: int
    warmup: Probe
    cold: Probe
    hit: Probe
    decode: Probe
    decode_code: Optional[Probe] = None   # None: a result from before the code probe existed

    @property
    def cold_was_cold(self) -> bool:
        """False when the seed was reused and the 'cold' prompt came back from the cache.
        A few cached tokens are still cold: the turn header before the nonce is the same in every
        prompt and some families' caches can hand back that much (Gemma: 10 tokens, exp05); the
        cache works in 256-token blocks, so less than one block means nothing after the header matched."""
        return self.cold.cached_tokens < 256

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
        return _decode_rate(self.decode)

    @property
    def decode_code_tok_s(self) -> Optional[float]:
        return None if self.decode_code is None else _decode_rate(self.decode_code)


def _decode_rate(p: Probe) -> float:
    return p.completion_tokens / max(1, p.total_ms - p.first_token_ms) * 1000


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
    lmk = usage_chunk.get("lmk") or {}
    return Probe(prompt_tokens=usage["prompt_tokens"],
                 cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                 completion_tokens=usage["completion_tokens"], first_token_ms=first, total_ms=chunks[-1][0],
                 restore_ms=lmk.get("restore_ms"), draft_accepted=lmk.get("draft_accepted"), draft_drafted=lmk.get("draft_drafted"))


def run_bench(stream: StreamFn, model_id: str, seed: Optional[int] = None,
              say: Callable[[str], None] = lambda _: None) -> BenchResult:
    seed = new_seed() if seed is None else seed
    warmup = _probe(stream, model_id, [{"role": "user", "content": "Reply with the single word: ready"}], 8)
    prompt = [{"role": "user", "content": probe_prompt(nonce_for(seed))}]
    cold = _probe(stream, model_id, prompt, 32)
    hit = _probe(stream, model_id, prompt, 32)
    decode = _probe(stream, model_id, [{"role": "user", "content": DECODE_PROMPT}], DECODE_TOKENS)
    decode_code = _probe(stream, model_id, [{"role": "user", "content": DECODE_CODE_PROMPT}], DECODE_TOKENS)
    return BenchResult(seed=seed, warmup=warmup, cold=cold, hit=hit, decode=decode, decode_code=decode_code)


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


def settings_note(model: dict) -> str:
    """The non-default switches a row was measured with, for the model cell: '' at the defaults."""
    parts = []
    if model.get("kv_cache_bits", 16) != 16:
        parts.append(f"KV cache {model['kv_cache_bits']}-bit")
    if model.get("speculative_decoding"):
        parts.append("speculative decoding")
    return f" ({', '.join(parts)})" if parts else ""


def _accepted(p: Optional[Probe]) -> str:
    return "" if p is None or p.acceptance is None else f" ({p.acceptance:.0%} of drafted tokens accepted)"


def markdown_row(r: BenchResult, m: dict, status: dict, date: str) -> str:
    model = status["model"]
    mem = f"{m['memory_gb']} GB" if m.get("memory_gb") else "?"
    cold = (f"{r.cold_prefill_tok_s:.0f} tok/s ({r.cold.prompt_tokens - r.cold.cached_tokens:,} tokens)" if r.cold_was_cold
            else f"— (seed reused: {r.cold.cached_tokens:,} cached)")
    decode = f"{r.decode_tok_s:.1f} tok/s"
    if r.decode_code is not None:
        decode = f"{r.decode_tok_s:.1f} tok/s prose · {r.decode_code_tok_s:.1f} code{_accepted(r.decode_code)}"
    return (f"| {m['chip']} | {mem} | {model['id']}{settings_note(model)} | {model['context_length']:,} | {cold} | "
            f"{_k(r.hit_tok_s)} tok/s ({r.hit.cached_tokens:,} cached; first token {r.hit_first_token_s:.2f} s) | {decode} | "
            f"{status.get('build', '?')} | {str(status.get('engine', '?'))[:7]} | {date} |")


def _k(tok_s: Optional[float]) -> str:
    return "?" if tok_s is None else f"{tok_s / 1000:.0f}k"


def human_block(r: BenchResult) -> str:
    cold = (f"{r.cold_prefill_tok_s:>7.0f} tokens/s" if r.cold_was_cold
            else f"      —           (seed {r.seed} reused and the cache still had it)")
    hit = f"{_k(r.hit_tok_s):>7} tokens/s" if r.hit_tok_s is not None else "       ? tokens/s   (this lmk does not report restore time)"
    lines = [f"  prefill          {cold}",
             f"  cached prefill   {hit}   ({r.hit.cached_tokens:,} of {r.hit.prompt_tokens:,} from disk; "
             f"first token {r.hit_first_token_s:.2f} s incl. the last partial block)",
             f"  decode, prose    {r.decode_tok_s:>7.1f} tokens/s" + _accepted(r.decode)]
    if r.decode_code is not None:
        lines.append(f"  decode, code     {r.decode_code_tok_s:>7.1f} tokens/s" + _accepted(r.decode_code))
    if r.warmup.first_token_ms > 3000:
        lines.append(f"  (the warm-up request took {r.warmup.first_token_ms / 1000:.0f} s: the weights had to be paged back in; not counted)")
    return "\n".join(lines)
