# exp05 — the smallest catalog model at chunk 2048 vs 512 (gemma-4-E4B-it 4-bit)

Written before the run (2026-11-01, 19:30). exp02/03 found chunk 512 free on
the 27B. A flat 512 default also applies to the small models in the catalog,
where the gist saw a cost (+21% prefill time on a 3B at 512).

## Purpose

Does chunk 512 slow prefill on a ~4B-class model enough to matter, and does
it still save memory there? This decides whether the default is a flat 512 or
needs a size rule.

## Method

`../shared/bench_model.py` — exp02's bench with the model path as an
argument — on `lmstudio-community/gemma-4-E4B-it-MLX-4bit` (the `lmk pull`
copy in the HF cache), fp16 cache, batched kit, `max_seq_nums=2`,
`max_kv_size=65536`, prompts 4k / 16k / 30k / 50k (its context is 131k),
greedy 64 tokens, peak per request. Arms `e4b-kv16-step2048`,
`e4b-kv16-step512`. Gemma 4's sliding layers use a 512-token window
(`sliding_window: 512`, `RotatingKVCache(keep=0)`), its full-attention layers
a plain `KVCache`; the engine log says whether it prefills in chunks at all
(`_requires_global_no_chunked_prefill`).

## Expectation

- **prefill time**: 512 slower than 2048 by 10–25% at 16k–50k (page: 2,199
  tok/s cold, so GEMMs at 512 rows are small for this GPU). Confidence:
  medium. At 4k the difference may be within noise.
- **peak memory**: a smaller saving than the 27B's, 0.5–1.5 GB at 50k (8
  attention heads vs the 27B's; the sliding layers' score matrix is bounded by
  the 512 window anyway). Confidence: medium.
- **decode**: unchanged.

Reading rule: if 512 costs ≥ 10% prefill time on this model, a flat 512
default is wrong for the catalog's small end — propose 1024 flat, or 512
only for models above some size; if ≤ 5%, flat 512.

## Baseline

`docs/models/gemma-4-e4b-4bit.md`: prefill 2,199 tok/s (cold), decode 93.5
tok/s, measured through lmk at chunk 2048. exp02/03 for the 27B.

## Run

```
research/2026-11-01-sink-mask-review/exp05-small-model-prefill-chunk/run.sh
```

## Result (2026-11-01 19:25, single run per arm, `raw/*.jsonl`)

| arm | peak GB 4k / 16k / 30k / 50k | prefill s 4k / 16k / 30k / 50k | prefill tok/s 4k / 16k / 30k / 50k | decode tok/s 50k |
|---|---|---|---|---|
| e4b @ 2048 | 7.4 / 8.1 / 9.1 / 10.7 | 1.9 / 6.1 / 11.3 / 20.1 | 2086 / 2628 / 2646 / 2488 | 49 |
| e4b @ 512 | 7.1 / 7.7 / 8.6 / 9.5 | 1.7 / 6.5 / 12.6 / 22.0 | 2422 / 2466 / 2388 / 2277 | 49 |

Prefill 512 vs 2048: +7% at 16k, +11% at 30k, +9% at 50k (4k is faster at
512, within noise). Peak −1.1 GB at 50k. Decode unchanged. Both within the
expectation's ranges, at the low end of the time cost. The reading rule's
10% line is hit exactly; the design doc takes flat 512 anyway and records
1024 and a size rule as rejected (the absolute cost is 1.9 s once per
conversation at 50k). Gemma 4 prefills in chunks on this kit (the log shows
a per-request plan at the configured step).
