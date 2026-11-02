# exp04 — how much a decoding request stalls while another reads a 30k prompt, chunk 2048 vs 512

Written before the run (2026-11-01, 19:20). The candidate headline benefit of
a smaller prefill chunk is not memory (exp02/03) but the second agent's
experience: CLAUDE.md records that reading a long prompt drags every
generating request to a near-stop, because each engine loop is "one token for
everyone plus one prefill chunk".

## Purpose

With two requests in flight on the 27B (lmk's `max_parallel: 2`): while B's
30k prompt is being read, how long are the gaps between A's tokens, at chunk
2048 and at chunk 512? And does A's decoding alongside cost B's prefill
anything (compare exp02's solo 102 s at 30k)?

## Method

`../shared/bench_stall.py`, one process per arm (`run.sh`), fp16 cache (the
owner's configuration; the chunk effect was the same for kv8 in exp02/03):
load as lmk does, warm up with one short request, then A (short prompt,
greedy, unlimited) decodes; once A has 10 tokens, B (30k filler prompt,
`max_tokens=4`) is submitted on a second thread. Every A token's arrival time
is recorded; A is stopped when B's first token arrives. Reported: median
inter-token gap of A before B started and during B's prefill, the max gap,
A's tokens per second during the window, B's prefill seconds. Raw A
timestamps are kept in the jsonl.

## Expectation

- **Chunk 2048**: A gets about one token per engine loop, and a loop is one
  2048-token chunk at the cold prefill rate (~300 tok/s, exp02) ≈ 6.8 s, so
  A's median gap during B's prefill ≈ 7 s (vs ~27 ms alone), i.e. ~0.15
  tok/s — "near-stop", as CLAUDE.md says. Confidence: medium-high; the one
  thing I have not read is whether the batched generator runs a decode step
  for A after *every* chunk or only after a whole prompt.
- **Chunk 512**: four times the loops, so a gap ≈ 1.7 s and ~0.6 tok/s.
  Confidence: medium-high, same caveat.
- **B's prefill**: a few percent slower than solo (102 s) because A's decode
  steps are interleaved; the 512 arm pays ~4× as many decode steps, so maybe
  +5–10 s there. Confidence: medium.

Reading rule: if 512 cuts A's median gap by ≥ 3× with ≤ 10% added prefill
time for B, that is the benefit the design doc leads with; if A gets no
tokens at all during prefill in either arm (the generator does not interleave
per chunk), the chunk size does not affect the stall and the design doc says
so — memory is then the only benefit.

## Baseline

exp02 `kv16-step2048` / exp03 `kv16-step512`: B-shaped prompt of 30k solo
read in 102 s / 106 s. No earlier measurement of inter-token gaps under
concurrent prefill; `docs/design/2026-09-20-memory-guard.md` records the
symptom ("reading 13k took 43 s, the other request produced 11 chunks").

## Run

```
research/2026-11-01-sink-mask-review/exp04-decode-stall-during-prefill/run.sh
```

## Result (2026-11-01 19:23, single run per arm, `raw/*.jsonl`; A's timestamps in the second row)

| arm | A alone, gap | A during B's prefill: tokens / median gap / max gap / tok/s | B prefill s (solo, exp02/03) |
|---|---|---|---|
| chunk 2048 | 25 ms | 18 / 6.7 s / 7.6 s / 0.18 | 102.2 (102) |
| chunk 512 | 25 ms | 62 / 1.8 s / 2.1 s / 0.58 | 106.3 (106) |

As expected: the generator does interleave one decode step per prefill
chunk, so A gets exactly one token per chunk; the gap is the chunk's prefill
time (6.2 s at the start of the 30k prompt, 7.6 s near the end, as attention
over the growing context costs more). 512 cuts the gap 3.7× (6.7 → 1.8 s).
B's prefill costs the same as solo in both arms, so A's interleaved decode
steps are free for B; the 512 arm's +4 s is the chunk size itself (exp03).
Reading rule met. A at 1.8 s per token is still far from its 25 ms alone —
512 turns a freeze into a trickle, not into normal speed.
