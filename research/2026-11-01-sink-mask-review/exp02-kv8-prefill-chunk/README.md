# exp02 — peak memory and prefill time vs prefill chunk, batched kv8 path, Qwen3.8-27B-4bit

Written before the run (2026-11-01 evening). Question from SMR-007: the fork's
chunk-512 default (branch `kv-quant-chunk`, `09b7473`) is justified by numbers
from mlx-lm's convert path on 1.5B–3.8B text models; what does it do on the
path lmk actually runs?

## Purpose

For the 27B lmk ships, with the KV cache at 8 bits on the batched kit: how much
does the per-process Metal peak drop when the prefill chunk goes 2048 → 1024 →
512, and what does prefill time pay for it? Plus an fp16-cache arm at 2048 so
the "quantized KV peaks above fp16" claim can be checked on this path.

## Method

`bench.py`, one process per arm (`run.sh`): `load_model(<catalog 27B-4bit>,
max_kv_size=65536, max_seq_nums=2, kv_bits=8|None, prefill_step_size=S,
enable_disk_cache=False)` — the batched kit, auto context fit on, lmk's
buffer-cache limit — then for prompt lengths 4k / 16k / 30k / 50k tokens
(filler sentences + one question, thinking off), one greedy request of 64
tokens. Per request: `mx.reset_peak_memory()` before, `mx.get_peak_memory()`
after; prefill seconds = time to first token; decode tok/s over the 64.
Machine: M3 Ultra 96 GB, resident lmk service idle (holds 17 GB, separate
process; peak is per process). Engine at fork `lmk` = `c3ee66f` (sink mask
only; the chunk is passed explicitly, so the held commit is not needed).
Disk cache off and no draft model: this isolates the chunk effect; the user's
configuration adds both.

Arms: `kv16-step2048`, `kv8-step2048`, `kv8-step1024`, `kv8-step512`.

## Expectation

- **kv8 @ 2048 vs fp16 @ 2048, peak**: *not* 66–73% above fp16. That figure is
  the convert path's, where fp16 and quantized caches coexist; here the
  quantized cache is built from token one. I expect kv8 peak within ±10% of
  fp16 at 30k, possibly below it at 50k (smaller cache). Confidence: medium.
- **kv8 @ 512 vs kv8 @ 2048, peak**: lower by the transient the gist
  describes (chunk × context score matrix, dequantized cache). At 50k I
  expect 1–3 GB less; at 4k, no visible difference. Confidence: medium-low
  (the fork computes attention through mlx-vlm's quantized SDPA; whether it
  materialises the full score matrix per chunk I have not read).
- **prefill time**: 512 slower than 2048 by 5–25% on the 50k prompt (smaller
  GEMMs, four times the loop overhead); 1024 in between. Confidence: medium.
  The gist saw "same prefill time" on a 1.5B and +21% on a 3B; a 27B's GEMMs
  are large enough at 512 that I lean to the low end.
- **auto context fit**: it may already pick a step below 2048 on its own when
  memory is tight (`PREFILL_STEP_CANDIDATES`); the log line at load says
  which. On 96 GB with kv8 I expect it to keep 2048.

Reading rule, set now: adopt 512 as the kv8 default only if the 50k peak drops
by ≥ 2 GB *and* 50k prefill slows by ≤ 10%; if the peak drop is real but the
slowdown is larger, propose 1024 or a size-dependent rule; if the peak barely
moves, drop the fork branch.

## Baseline

No earlier lmk measurement of peak memory by chunk size. Related:
`research/2026-09-23-kv-cache-quant/` exp01 measured per-token cache bytes
(kv8 = 34,816 B/token on this model) and throughput, not peak.
`docs/benchmarks.md` numbers were taken at 2048.

## Run

```
research/2026-11-01-sink-mask-review/exp02-kv8-prefill-chunk/run.sh
```
Raw: `raw/<arm>.jsonl` (one row per prompt length, first row = load info),
`raw/<arm>.log` (engine log), `raw/run.log` (arm order, exit codes, times).
