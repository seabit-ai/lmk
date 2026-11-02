# exp03 — the fp16-cache arm at chunk 512 (exp02 only had fp16 at 2048)

Written before the run (2026-11-01, 18:30). exp02 showed kv8 @ 512 peaks 4.7 GB
below kv8 @ 2048 at 50k with no prefill-time cost. The fork's rule lowers the
chunk only when the cache is quantized. Whether that condition is the right
one depends on this arm.

## Purpose

Does the same chunk-size saving appear with the fp16 cache? If yes, the
transient is the chunk × context score matrix regardless of KV bits, and the
default should drop to 512 for every load, not just quantized ones.

## Method

Identical to exp02 (`../shared/bench.py`, a verbatim copy of exp02's
`bench.py`), one arm: `kv16-step512`. Same machine state (M3 Ultra 96 GB,
resident lmk idle in its own process), same prompts 4k / 16k / 30k / 50k,
greedy 64 tokens, peak per request.

## Expectation

- 50k peak: 25.5 GB (exp02 fp16 @ 2048) → about 21 GB, i.e. a 4–5 GB drop
  like kv8's. Confidence: medium-high — the per-chunk transient should not
  depend on how the cache is stored.
- Prefill time within 3% of exp02's fp16 @ 2048 (187 s at 50k), as it was
  for kv8. Confidence: high.
- Decode unchanged (30 tok/s at 50k).

Reading rule: if the 50k drop is ≥ 2 GB with ≤ 10% prefill cost, propose
"512 for every load" instead of the fork's "512 when quantized"; if fp16 does
not move, the fork's condition is right as it stands.

## Baseline

exp02, arm `kv16-step2048`: peak 18.1 / 19.9 / 22.2 / 25.5 GB, prefill
13 / 52 / 102 / 187 s, decode 38 / 35 / 33 / 30 tok/s.

## Run

```
research/2026-11-01-sink-mask-review/exp03-fp16-prefill-chunk/run.sh
```
