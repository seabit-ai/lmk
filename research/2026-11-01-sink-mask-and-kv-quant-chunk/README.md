# Sink-mask fix + KV-quant prefill chunk size — 2026-11-01

Two small engine-side fixes for `RotatingKVCache` attention-sink recall and
Metal peak memory under quantized KV prefill. Bug report and fix direction
credited to **Robert Price (blackweb.ai)**, upstream issue
[ml-explore/mlx-lm#1978](https://github.com/ml-explore/mlx-lm/issues/1978).

> **⚠️ LATE FINDING (17:18 PDT 11-01, after the needle A/B run): the sink patch
> is wired on the WRONG kit for lmk's production loads.** Details in §7.
> In one sentence: `MlxEngine` always loads `max_seq_nums=2` → BatchedVisionModelKit
> (batched path), but `apply_rotating_sink_mask_patch` is only called in
> `cache_wrapper._make_cache` — the sequential (ModelKit) path. The batched
> path never sees the patch. The needle tests that "passed" ran unpatched and
> passed because Qwen3.5-27B is sink-insensitive on this path (same as Phi in
> Robert's grid), NOT because the fix is active there.
>
> State of things as of this note: sequential path = fixed and verified (unit
> + semantics + itest). Batched path = NOT fixed yet, needle test has zero
> discriminating power on this model, A/B variant still fails with
> AttributeError (no cache_wrapper on the batched kit). Next step: wire the
> patch into `kv_quant.make_prompt_cache_for` (one line) or
> `batch_generator.py:863`, fix the A/B to find the batch generator's
> per-batch cache list, re-run. §7 has the full call-chain evidence.

> Reviewed in `research/2026-11-01-sink-mask-review/` (2026-11-01 evening): the conclusions above and in §7 do not hold for lmk's catalog. Read that first.

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

Unit tests (`tests/test_rotating_sink_mask.py`, no real model weights):
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
256 → 256), and (11) the full semantics re-run against the mlx-vlm copy
(`test_semantics_on_mlx_vlm_copy`) — the vlm class had only usage/idempotence
checks before this.

End-to-end (real model, `tests/test_itest_chat.py::
test_needle_recalls_across_a_bounded_window`, `make itest-needle`): a needle
planted in the system block, filler large enough to exceed the 32768-token
bounded window (chunked prefill runs many chunks and the cache trims), greedy
answer must contain the access code. Bobby Price's probe measured 0/8 recall
unpatched vs 8/8 patched; this test pins the patched behavior so a future
make_mask regression cannot pass CI silently. (Unpatched, the same prompt
fails — the bug demonstrably bites through the full lmk surface.)

Also: `make test` 270 passed / 13 skipped; `make itest` 10 passed / 3 skipped
(no regressions across tool-round-trip, speculative decoding, seed replay and
cache-restore acceptance); `make lint` clean.

## 5. Not verified (left to owner)

- Metal peak-memory re-measurement with kv8 at chunk 512 vs 2048 — a
  measurement, not an assertion (±30% run-to-run drift), belongs in a
  research exp, not in CI.
- The batched-vision path (`BatchRotatingKVCache`, `ml-explore` batch
  generator) was left untouched — it only serves models without `.merge`
  and is not on lmk's quantized/sequential path; its `make_mask` has a
  different (left_padding-aware) shape and would need its own analysis.

## 6. Files

- New: `mlx_engine/model_kit/patches/rotating_sink_mask.py`
- New: `tests/test_rotating_sink_mask.py` (11 tests)
- Modified: `mlx_engine/cache_wrapper.py` (patch wiring +
  `lower_prefill_chunk_for_kv_quant`)
- Modified: `mlx_engine/generate.py` (chunk-size call site)
- Modified (lmk): `tests/test_itest_chat.py` (needle-recall itest)
- Modified (lmk): `Makefile` (`engine-sink-test`, `itest-needle` targets)

Commits: engine fork branch `lmk` (owner), lmk main (research notes + targets).

## 7. LATE FINDING: the patch misses lmk's production path (2026-11-01 17:18 PDT)

Discovered while writing the unpatched A/B of the needle test: the variant
could not find a `cache_wrapper` on the kit (`'BatchedVisionModelKit' object
has no attribute 'cache_wrapper'`). The full call-chain audit that followed:

**How lmk loads the engine** (`lmk/engine.py:133`):
```
MlxEngine.__init__ → load_model(model_path, max_kv_size=requested,
                                max_seq_nums=max_parallel(=2), kv_bits=…)
```
`mlx_engine/generate.py` load_model: `max_seq_nums=2` → `parallel_requested` →
**BatchedVisionModelKit** (batched path) for text-only models that support
continuous batching (cache classes have `.merge`). ModelKit (sequential) is
only taken by: `max_seq_nums == 1` (escape hatch), kv_bits is not None (kv
quantization), vocab_only, or a non-batchable architecture.

**Where the patch is wired**: only `cache_wrapper.py:98` (`_make_cache`) —
that class belongs to ModelKit/CacheWrapper (sequential). The batched path
builds caches in `batch_generator.py:863` via `kv_quant.make_prompt_cache_for`
(`model_kit/batched_vision/kv_quant.py:29`) → `mlx_vlm.make_prompt_cache`
(`site-packages/mlx_vlm/models/cache.py:45`) → `RotatingKVCache(max_size, keep=4)`
— **never passing through the patch**.

Consequences, in order of importance:

1. **lmk's default production load is on the batched path, so the sink-mask
   fix is currently INERT for it.** The itest needle runs (2× passed, ~2m14s
   each) ran on an UNPATCHED cache and passed — Qwen3.5-27B recalls across a
   bounded window on this path regardless of the sink columns. Robert's grid
   showed the sensitivity is model-dependent (Qwen2.5-1.5B and Llama-3.2-3B
   collapse, Phi-3.5 does not); 27B behaves like Phi here.
2. The sequential path IS fixed and covered (unit tests 11/11, semantics,
   itest no-regression, itest-needle) — that path serves the kv8 + speculative
   workloads (exp05) and the `max_seq_nums=1` escape hatch.
3. The unpatched A/B itest (`test_needle_recalls_across_a_bounded_window_
   unpatched`) is currently a dud twice over: it AttributeError's (no
   cache_wrapper on the batched kit) and even once fixed it will likely PASS
   on 27B (insensitive), so its "must fail" assertion will trip the OTHER way.

**Fix direction (next session or now, ~10 min + one itest re-run)**:
- Wire `apply_rotating_sink_mask_patch` into the batched path — single call
  site: `batch_generator.py:863` (`self.prompt_cache = make_prompt_cache_for(...)`
  → wrap the result) or inside `kv_quant.make_prompt_cache_for` itself
  (covers context_fit.py:529's probe path too, which also builds caches).
- Re-audit the A/B: the batched kit's caches live per-batch
  (`_PromptPrefill.prompt_cache` at batch_generator.py:863, handed by reference
  to GenerationBatch.prompt_cache at :1138). Either wrap
  `apply_rotating_sink_mask_patch` in the test to record every cache list it
  sees and unpatch those, or expose a test-only hook on the kit.
- Discriminating power on 27B will likely stay absent (model-insensitive) —
  options: (a) assert A/B only when the probe's sensitivity is confirmed (run
  the needle on a sink-sensitive small model, e.g. Qwen2.5-1.5B-4bit — cheap
  itest), or (b) keep the A/B unpatched-fails expectation but mark it
  xfail-with-note, or (c) drop the itest-level A/B and rely on the mask
  semantics unit tests (they pin the bug directly). Owner's call.
- Also worth auditing: does the batched path's prefill chunking interact with
  the quantized-KV chunk lowering the same way? `PREFILL_STEP_CANDIDATES =
  (2048, 1024, 512)` in context_fit.py — the lowering only fires on the
  sequential path (`generate.py:242` clamps `prefill_step_size` before the kit
  branches, so actually BOTH kits get the clamped value — verify).

**Honest status table**:

| path | used by lmk for | sink patch | verified |
|---|---|---|---|
| sequential (ModelKit) | kv8 loads, speculative decoding, max_seq_nums=1 | ✅ wired | unit 11/11 + itest no-regression |
| batched (BatchedVisionModelKit) | **default loads (max_seq_nums=2)** | ❌ NOT wired | needle "passed" = model insensitive, not the fix |

