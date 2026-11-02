# Sink-mask fix + KV-quant prefill chunk size — 2026-11-01

Two small engine-side fixes for `RotatingKVCache` attention-sink recall and
Metal peak memory under quantized KV prefill. Bug report and fix direction
credited to **Robert Price (blackweb.ai)**, upstream issue
[ml-explore/mlx-lm#1978](https://github.com/ml-explore/mlx-lm/issues/1978).

## 1. What broke

Upstream `RotatingKVCache.make_mask` (the `N > 1` branch, identical in
`mlx_lm.models.cache` and `mlx_vlm.models.cache`) hides the attention-sink
(`keep`) columns from chunked-prefill queries once the cache starts slicing
(`offset >= max_size`): the sliding-window cut is applied to a key count that
includes the whole current chunk, so the sink columns at the buffer head fall
outside `window_size`. A single-token decode query at the same position sees
the whole buffer (`mask is None` when the window covers the cache), so:

- chunked prefill: sinks invisible → degraded recall for system-prompt-like
  tokens in long prefills,
- decode: sinks visible → the two paths disagree on the visible column set.

## 2. The fix

New file `mlx_engine/model_kit/patches/rotating_sink_mask.py`
(`apply_rotating_sink_mask_patch`), wired in `cache_wrapper._make_cache` —
every prompt cache the sequential path builds (i.e. every cache that passes
`max_kv_size` to `make_prompt_cache`) gets a per-instance patched `make_mask`
(MethodType, idempotent, matching the `gemma4.py` patch pattern). Both
`mlx_lm` and `mlx_vlm` `RotatingKVCache` instances are covered by one
implementation (matched by class name + `make_mask/max_size/offset/keep`
attrs — the two libraries ship byte-identical buggy `make_mask` bodies).

Mask semantics for multi-token prefill (`N > 1`, `window_size >= max_size`):
key count = `min(max_size - 1, offset) + N` (upstream's own convention — the
chunk being prefilled is part of the buffer this mask describes, and query
rows are labelled from `max_size - 1` when slicing so the causal band stays
exact). Column rule per row: causal **and** (`position < query + (max_size -
keep)` **or** `position < keep`):

- sink columns stay visible to every prefill row (causal always allows it:
  every prefill row sits past `max_size - 1` while sinks label `0..keep-1`),
- the sliding part shrinks from `max_size` to `max_size - keep` to pay for
  the sinks,
- so each prefill row sees exactly the column set the matching decode query
  sees (`keep` sinks + the `max_size - keep` most recent tokens).

Deliberately left upstream: `N == 1` (decode) and windows smaller than the
cache — a sub-cache-size window hides the sinks in decode too, so upstream
semantics are already self-consistent there.

## 3. Chunk size under quantized KV

`cache_wrapper.lower_prefill_chunk_for_kv_quant(prefill_step_size, kv_bits)`:
when KV quantization is on and `prefill_step_size` was **not** explicitly
requested, the default chunk drops 2048 → 512 (quantized prefill allocates
the full quantized block per chunk; the 2048 default spikes Metal peak
memory — Robert Price measured ~52% savings at 512). Explicit wins.
Wired at `generate.py`'s single `validate_prefill_step_size` call site
(`lmk` never passes `prefill_step_size`, so every quantized load takes 512).

## 4. How it was verified

Unit tests only (`tests/test_rotating_sink_mask.py`, no real model weights):
synthetic chunks into a real `RotatingKVCache(max_size=16, keep=4)` from both
libraries, asserting (1) sink columns visible to every prefill row once
slicing starts, (2) every prefill row sees exactly `keep + (max_size - keep)`
columns, (3) the last prefill row's visible column set equals what a
single-token decode at the same offset sees, (4) sink slots in the buffer
actually survive the update (keep-position invariance), (5) the bug is pinned
on unpatched upstream (sinks dropped), (6) small prefills still return the
`"causal"` string fast path and decode stays `None`, (7) sub-window sizes are
left untouched, (8) patch is idempotent and applies to both libraries'
copies, (9) non-rotating caches in a list are untouched, (10) the chunk-size
rule's three cases (no kv_bits → 2048; kv_bits=8 default → 512; explicit
256 → 256).

Run:

```
cd .engine/mlx-engine && PYTHONPATH=. ../../.venv/bin/python -m pytest \
  tests/test_rotating_sink_mask.py -x -q
```

## 5. Not verified (left to owner)

- Real-model needle/recall test on a sliding-window model (exp-style, e.g.
  Qwen3.5/3.8 long-session recall with `max_kv_size` set) — the unit tests
  pin mask semantics, not end-to-end model behavior.
- Metal peak-memory re-measurement with kv8 at chunk 512 vs 2048.
- The batched-vision path (`BatchRotatingKVCache`, `ml-explore` batch
  generator) was left untouched — it only serves models without `.merge`
  and is not on lmk's quantized/sequential path; its `make_mask` has a
  different (left_padding-aware) shape and would need its own analysis.
- Not committed (fork branch + `ENGINE_COMMIT` + owner push flow is the
  owner's call per CLAUDE.md git rules).

## Files

- New: `mlx_engine/model_kit/patches/rotating_sink_mask.py`
- New: `tests/test_rotating_sink_mask.py`
- Modified: `mlx_engine/cache_wrapper.py` (patch wiring +
  `lower_prefill_chunk_for_kv_quant`)
- Modified: `mlx_engine/generate.py` (chunk-size call site)
