# Review of the sink-mask / KV-quant-chunk work — 2026-11-01 (evening)

A second agent's review of `research/2026-11-01-sink-mask-and-kv-quant-chunk/`
(written earlier the same day by another agent, kept as it was) and of the
fork commit `44fb344` it describes. Every claim below was checked against the
code in `.venv` (mlx-lm / mlx-vlm), the fork, and a scratch sweep; the source
email was read with `kitten-gmail`.

**Verdict in one line:** the mask patch is mathematically correct, and a no-op
for every model lmk ships, on both engine paths; the chunk-512 change is the
one with real blast radius and it is unmeasured on lmk's path.

## SMR-001 Where the work came from

An email from Robert "Bobby" Price (blackweb.ai), 2026-10-17, subject
"exp05-kv8-with-draft/smoke.py in seabit-ai/lmk + MLX KV cache bugs". A cold
business outreach ("Business inquiry. Reply 'no thanks' and I will not write
again"; asks for a 10-minute chat), triggered by reading
`research/2026-09-23-speculative-decoding/exp05-kv8-with-draft/smoke.py`
(`load_model(..., max_kv_size=32768, max_seq_nums=2, kv_bits=KV_BITS)`). Not a
report of anything observed in lmk; names no model. Two claims, each linked:

1. `RotatingKVCache` trims during chunked prefill mask out the 4 `keep` sink
   columns; "in-window recall drops from 8/8 to 0/8" —
   [mlx-lm PR #1978](https://github.com/ml-explore/mlx-lm/pull/1978)
   (a PR, not an issue; no maintainer response as of 2026-11-01; one reviewer
   reported a failing shape for the PR's own implementation).
2. Quantized KV at the 2048 prefill step "spikes peak Metal RAM 66%–73% above
   FP16" on 16k–32k prompts; step 512 "cuts peak RAM by up to 52%" —
   [gist 88f542f](https://gist.github.com/robertcprice/88f542f29ec8b97f2a8b1edde3fdc903).

All numbers and the per-model recall table in the other notes come from the
gist. Its conditions: mlx-lm 0.31.3 `stream_generate` directly (no
mlx-engine, no batching); Qwen2.5-1.5B, Llama-3.2-3B, Phi-3.5-mini, all 4-bit
text-only; prompts 512 / 4k / 16k / 30k. The memory spike is on mlx-lm's
**convert path** (`kv_bits` turns an fp16 cache into `QuantizedKVCache`
mid-generation, both alive at once). The gist lists a "direct construction"
path but publishes no numbers for it. The mechanism it gives for the chunk
effect — prefill attention dequantizes the cache into a chunk × context score
matrix — does apply to lmk's path.

The exp05 trigger is moot: that smoke runs Qwen3.5, which ignores
`max_kv_size` (SMR-002).

## SMR-002 No model lmk ships ever builds the cache the bug lives in

`make_prompt_cache(model, max_kv_size)` (`mlx_lm/models/cache.py:15`,
`mlx_vlm/models/cache.py:45`) builds `RotatingKVCache(max_kv_size, keep=4)`
**only when the model has no `make_cache`**. Every `TESTED_MODELS` family has one:

| family | `make_cache` returns | bug applies? |
|---|---|---|
| Qwen3.5 / 3.8, 27B and 122B (`mlx_vlm/models/qwen3_5/language.py:2204`; MoE inherits) | `ArraysCache(size=2)` on linear layers, `KVCache()` on full-attention layers — no window | no: no rotating cache exists |
| Gemma 4 (`mlx_vlm/models/gemma4/language.py:859`) | `KVCache()` / `RotatingKVCache(sliding_window, keep=0)` | no: with `keep=0` the patched mask equals upstream's |

So the needle itest in the other notes passed because the prompt was never
windowed — `max_kv_size=32768` is ignored by Qwen3.5 on both paths — not
because "27B is sink-insensitive like Phi". Patched and unpatched runs are
the same computation on the catalog; an unpatched-must-fail assertion can
never hold there.

## SMR-003 lmk's catalog never takes the sequential path

`generate.py` `load_model` branches on `"vision_config" in config_json`
before it looks at `max_seq_nums` or `kv_bits`. Every listed model has
`vision_config` (including the owner's text-only checkpoint of a vision
architecture), so `BatchedVisionModelKit` serves kv8, speculative decoding
and `max_parallel: 1` alike. The other notes' table row "sequential — used by
lmk for kv8 loads, speculative decoding" is wrong for lmk; the sequential
path is reached only by a text-only model with no `vision_config`.

## SMR-004 The batched path cannot have this bug

`kv_quant.make_prompt_cache_for` calls `make_prompt_cache(model)` with no
`max_kv_size`; on the batched kit `max_kv_size` only sizes the prompt-cache
store (`batched_vision/model_kit.py:192`, `prompt_cache/cache_store.py:127`)
and the disk budget. The only rotating caches reaching that path are a
model's own (Gemma 4, `keep=0`; `context_fit.py:573` refuses `keep != 0`).
"Wire the patch into `make_prompt_cache_for`" — the other notes' next step —
would be a no-op for lmk.

## SMR-005 The patch itself is right

Beyond the fork's 11 unit tests (pass), a sweep over
`max_size ∈ {8,16,32}`, `keep ∈ {0,1,4,7}`, prefill chunk `∈ {3,9,16}`,
history `∈ {0,5,19,40}`, final chunk `∈ {2,10,33}` — mask built *before*
the update, as the forward pass does; keys carry their own position —
compared the last prefill row's visible positions with "the `keep` sinks plus
the `max_size − keep` most recent tokens" (what decode sees):

| | cases | array masks | mismatches |
|---|---|---|---|
| upstream | 432 | 288 | 216 (all `keep > 0`; sinks missing) |
| patched | 432 | 288 | 0 |

Mask width always matched the key count. The shape a reviewer reported
failing on the upstream PR's own implementation (`max_size=8, keep=7,
history=19, step=9, n=10`) is handled by ours: last row sees `[0..6, 28]`.
Script: `exp01-mask-sweep/sweep.py`, output in `exp01-mask-sweep/raw/`.

## SMR-006 Where the fix does matter

A user pointing `model.path` at a text-only model whose family has no
`make_cache` (Llama, Qwen2/Qwen3 dense, Mistral, Phi-3…) *and* landing on
the sequential path (`kv_cache_bits: 8|4`, or `max_parallel: 1`) gets
`RotatingKVCache(context_length, keep=4)` and, before the fix, lost the first
four tokens of every prompt longer than the context during prefill. That is
Robert's configuration, not lmk's catalog. A needle A/B can only be shown on
such a model (e.g. Qwen2.5-1.5B-4bit via `LMK_ITEST_MODEL` with
`LMK_ITEST_KV_BITS=8`).

## SMR-007 Chunk 512 has the wider blast radius, and no lmk measurement

`lower_prefill_chunk_for_kv_quant` runs at `generate.py:242`, before the kit
branch, so **both** kits get 512 whenever `kv_bits` is set — every lmk load
on a Mac under 96 GB (`kv_cache_bits: auto` → 8). Prefill then runs four
times as many engine loops per prompt (the admission queue's rule two exists
because each loop is one 2048 chunk). The up-to-52% peak saving is the
gist's number on mlx-lm's convert path with 1.5B–3.8B text models; neither
the saving nor the prefill-speed cost has been measured on lmk's batched kv8
path (`docs/benchmarks.md` and the model pages were measured at 2048).
Needs its own exp before it ships.

## SMR-008 Housekeeping on the branch

- Fork commit `44fb344` bundles two features (sink mask + chunk 512);
  CLAUDE.md asks for one commit per feature. Its
  `lower_prefill_chunk_for_kv_quant` docstring cites the PR for the memory
  finding; that finding is in the gist.
- Uncommitted in the fork: `remove_rotating_sink_mask_patch` + a vlm
  semantics test. Uncommitted in lmk: `itest-needle` target and two needle
  itests. All exist to serve an A/B that cannot discriminate on the catalog.
- `ENGINE_COMMIT` still `e74492f` (pre-fix). `make test` would check the
  engine worktree out to it; run as plain pytest instead: 270 passed /
  15 skipped, lint clean (2026-11-01).
- The other notes' §4 claims "Unpatched, the same prompt fails"; never
  observed (their §7 says the A/B errored). §5 says the batched path "only
  serves models without `.merge`" — backwards.

## SMR-009 The batched kit already plans its prefill chunk per request

`ContextFitResult.make_request_prefill_plan` (`batched_vision/context_fit.py:113`)
builds, for each prompt, a schedule of segments that step down through
`PREFILL_STEP_CANDIDATES = (2048, 1024, 512)` as the cache grows past what the
safe ceiling (free memory minus a 3 GiB reserve) allows; `_next_prompt_step_size`
in the batch generator follows it. So on a tight machine upstream mlx-engine
lowers the chunk by itself, per request, for both cache kinds. On the 96 GB
test machine the ceiling is 80 GB and every exp02 prompt ran as one 2048
segment. The fork's rule and this plan compose: a configured 512 leaves the
plan only `[512]`. The fork's rule is therefore about the *default* on
machines where the fit sees no pressure — exactly where exp02 measured.

## SMR-010 exp02: chunk 512 on lmk's batched kv8 path costs nothing and saves 2–5 GB

Qwen3.8-27B-4bit, M3 Ultra 96 GB, batched kit, one process per arm, greedy
64 tokens (`exp02-kv8-prefill-chunk/`, single run each):

| arm | peak GB 4k / 16k / 30k / 50k | prefill s 4k / 16k / 30k / 50k |
|---|---|---|
| kv16 @ 2048 | 18.1 / 19.9 / 22.2 / 25.5 | 13 / 52 / 102 / 187 |
| kv8 @ 2048 | 18.0 / 19.4 / 21.4 / 24.2 | 13 / 52 / 103 / 191 |
| kv8 @ 1024 | 16.9 / 17.9 / 19.3 / 20.9 | 13 / 52 / 103 / 187 |
| kv8 @ 512 | 16.4 / 17.4 / 18.7 / 19.5 | 13 / 54 / 106 / 192 |

- The gist's "+66–73% above fp16" does not happen here: kv8 peaks 1.3 GB
  *below* fp16 at 50k. Convert-path artefact, as SMR-001 predicted.
- 2048 → 512 under kv8: −4.7 GB at 50k, −1.6 GB already at 4k; prefill +0.5%
  at 50k; decode identical. My expectation undershot the saving (1–3 GB) and
  overshot the cost (5–25%).
- Cold prefill on this model is ~300 tok/s (model page: 323), so each arm
  took 6 minutes, not the 2 I had assumed when scheduling.

Not measured: fp16 at 512 (exp03), a memory-tight machine where the fit's own
plan (SMR-009) would already step down, the effect with the draft model and
disk cache on, run-to-run drift (one run per arm; the four arms agree to
0.1 GB where they should, which argues against the ±30% drift the earlier
notes assumed).

## SMR-011 exp03: the chunk saving is the same with the fp16 cache — the "when quantized" condition is wrong

fp16 cache, same bench, chunk 512 vs exp02's 2048: peak 16.5 / 17.5 / 18.9 /
20.8 GB vs 18.1 / 19.9 / 22.2 / 25.5 (−4.7 GB at 50k, the same as kv8's
−4.7), prefill 189 s vs 187 s at 50k, decode unchanged. The transient the
chunk size controls is the per-chunk attention working set, independent of
KV bits; the gist's own Phi/Llama fp16-at-512 columns show it too. So the
fork's rule (`lower_prefill_chunk_for_kv_quant`: 512 only when `kv_bits` is
set) keys on the wrong condition: on this path it would leave the 96 GB
machines (fp16 by default) at 2048 and the under-96 GB ones at 512, when
both gain the same.

Peak at 50k on the 27B, this machine, by chunk (GB): 2048 → 25.5 (fp16) /
24.2 (kv8); 1024 → 20.9 (kv8); 512 → 20.8 (fp16) / 19.5 (kv8).

## Recommendation

Keep the mask patch (correct, cheap, helps the `model.path` case in
SMR-006), as its own commit. Drop the two needle itests and the
`itest-needle` target, or gate them on a text-only `LMK_ITEST_MODEL`. The
chunk change passed the pre-set rule on lmk's path for both cache kinds
(SMR-010, SMR-011: −4.7 GB at 50k, ≤ 1% prefill cost, decode unchanged).
Proposal: make 512 the default prefill chunk for **every** load — one line,
`PROMPT_PROCESSING_CHUNK_SIZE` 2048 → 512 in the fork, with the exp IDs as its
provenance — and drop `lower_prefill_chunk_for_kv_quant` (branch
`kv-quant-chunk`) since its condition is wrong. A product default, so the
owner's call; not applied. Worth measuring after: how much a concurrent
decode stalls during another request's prefill at 512 vs 2048 (the admission
queue's rule two exists because of that stall), and the model pages' prefill
numbers at 512.
