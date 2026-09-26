## Requests (decode tok/s from LmkChatDone; rounds counted by the profiler)

| kind | sampling | profile | decode tok/s median (min–max) | rounds / request | round ms = decode ms / rounds | accepted / drafted |
|---|---|---|---|---|---|---|
| code | greedy | phases | 42.0 (42.0–42.1) | 66 | 91.9 | 570 / 687 (83%) |
| code | greedy | wall | 43.4 (41.0–43.4) | 66 | 89.1 | 570 / 687 (83%) |
| code | sampled | phases | 35.3 (35.2–38.2) | 91 | 77.7 | 495 / 675 (73%) |
| code | sampled | wall | 36.7 (36.6–39.3) | 84 | 81.1 | 513 / 685 (75%) |
| prose | greedy | phases | 33.3 (33.2–33.3) | 117 | 65.5 | 414 / 699 (59%) |
| prose | greedy | wall | 33.3 (31.7–34.0) | 117 | 65.4 | 414 / 699 (59%) |
| prose | sampled | phases | 29.0 (28.6–32.3) | 125 | 67.4 | 392 / 746 (53%) |
| prose | sampled | wall | 30.8 (27.3–30.9) | 125 | 68.9 | 393 / 747 (53%) |

## Wall profile: one round in the engine's own schedule (ms per round, mean)

| kind | sampling | rounds | verify positions | walked positions | step (next()) | gap between steps (median; the mean carries the prefill before round 1) | round | draft build | draft wait (1st sync) | verify build | verify wait | walk host (pos 1 build/tolist, lse build) | positions 2.. (build+eval+tolist) | bookkeeping | rollback build | rest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| code | greedy | 198 | 4.47 | 1.00 | 91.8 | 0.08 | 91.4 | 0.65 | 7.88 | 3.42 | 79.18 | 0.01 | 0.00 | 0.03 | 0.19 | 0.01 |
| code | sampled | 253 | 3.71 | 3.03 | 81.4 | 0.08 | 81.1 | 0.66 | 7.10 | 3.56 | 67.45 | 0.09 | 1.90 | 0.02 | 0.26 | 0.02 |
| prose | greedy | 351 | 2.99 | 1.00 | 66.6 | 0.08 | 66.2 | 0.64 | 6.29 | 3.65 | 55.25 | 0.01 | 0.00 | 0.03 | 0.33 | 0.01 |
| prose | sampled | 374 | 3.00 | 2.05 | 69.7 | 0.08 | 69.3 | 0.72 | 6.46 | 4.25 | 56.30 | 0.10 | 1.02 | 0.03 | 0.42 | 0.02 |

## Wall profile at the same verify width (the width decides the verify forward; compare greedy and sampled at equal width)

| kind | sampling | verify positions | rounds | round median | round mean | verify wait median | draft wait median | verify build median | positions 2.. mean | walked positions | host-slow rounds (round > median + 5 ms) | verify build in host-slow rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| code | greedy | 3 | 48 | 63.5 | 64.4 | 54.3 | 5.9 | 2.50 | 0.00 | 1.00 | 6 (12%) | 2.78 |
| code | greedy | 5 | 147 | 98.3 | 100.8 | 87.0 | 8.0 | 2.56 | 0.00 | 1.00 | 19 (13%) | 10.60 |
| code | sampled | 3 | 163 | 66.0 | 67.8 | 55.2 | 5.9 | 2.52 | 1.57 | 2.69 | 26 (16%) | 7.52 |
| code | sampled | 5 | 89 | 102.4 | 105.0 | 88.4 | 8.0 | 2.51 | 2.50 | 3.64 | 16 (18%) | 9.55 |
| prose | greedy | 3 | 348 | 64.6 | 66.3 | 55.4 | 6.0 | 2.54 | 0.00 | 1.00 | 72 (21%) | 7.58 |
| prose | sampled | 3 | 373 | 66.7 | 69.3 | 56.3 | 6.0 | 2.55 | 1.02 | 2.05 | 111 (30%) | 8.11 |

## Phases profile: a sync after every phase (ms per round, mean; each = that phase's GPU time + one sync)

| kind | sampling | rounds | draft | verify | argmax (greedy) | logsumexp (sampled) | per walked position: build / eval / tolist | walk total | rollback | round |
|---|---|---|---|---|---|---|---|---|---|---|
| code | greedy | 198 | 6.25 | 78.95 | 0.30 | 0.00 | — | 0.37 | 0.94 | 89.8 |
| code | sampled | 273 | 5.28 | 62.90 | 0.00 | 0.30 | 0.02 / 0.85 / 0.04 | 2.91 | 0.96 | 75.8 |
| prose | greedy | 351 | 4.73 | 54.82 | 0.30 | 0.00 | — | 0.37 | 1.17 | 64.4 |
| prose | sampled | 374 | 4.79 | 55.10 | 0.00 | 0.30 | 0.02 / 0.86 / 0.04 | 2.26 | 1.26 | 67.3 |

## Micro-benchmark on real 32k verify logits (ms, median of per-round medians; 5 reps each after a warm call)

| logits from | samples | positions | per_position_all | vectorized_all | argmax_block | lse_one | top_p_one | top_k_one | categorical_one | sync_only |
|---|---|---|---|---|---|---|---|---|---|---|
| code greedy | 18 | 3.00 | 2.68 | 1.21 | 0.25 | 0.23 | 0.52 | 0.44 | 0.34 | 0.00 |
| code sampled | 18 | 3.00 | 2.67 | 1.21 | 0.25 | 0.23 | 0.52 | 0.44 | 0.33 | 0.00 |
| prose greedy | 18 | 3.00 | 2.67 | 1.21 | 0.25 | 0.23 | 0.51 | 0.44 | 0.34 | 0.00 |
| prose sampled | 18 | 3.00 | 2.69 | 1.21 | 0.25 | 0.23 | 0.52 | 0.44 | 0.34 | 0.00 |

rounds the profiler could not time (logits processors or a positioned sampler): 0

per_position_all = the engine's walk over every verify position (logsumexp stack, then per position: sampler, mx.eval, tolist); vectorized_all = one logsumexp over the block, the sampler on [positions, vocab], one eval, one tolist; the *_one rows are the pieces for one position; sampler = temp 1.0 / top_p 0.95 / top_k 20.
