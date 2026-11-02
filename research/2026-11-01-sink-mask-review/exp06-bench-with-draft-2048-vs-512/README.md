# exp06 — `lmk bench` with the draft model on: engine chunk 2048 vs 512

Written before the run (2026-11-01, 19:55). After the 512 default was
deployed, the owner's `lmk bench` on the resident service (private 27B
variant, lmk's copy of the official head) showed prose decode 44.3 tok/s at
50% of drafted tokens accepted, code 53.7 at 71%. There is no earlier row for
that variant, and exp02–05 measured decode without a draft. The drafter runs
during decode only, so the chunk should not touch it — but the main usage is
decode with the draft, and SPD-022 says acceptance must be checked in the
shape users run (CLAUDE.md).

## Purpose

Same model (catalog 27B-4bit, its own MTP head), same config, same bench
prompts (`--seed 20261101`, fresh cache dir per arm so both are cold): does
the engine at chunk 512 (`e863778`) give the same decode speed and
acceptance as at 2048 (`c3ee66f`)? And a benchmarks.md row for the catalog
model at 512.

## Method

`run.sh`: per arm, a temporary lmk (`LMK_HOME` in /tmp, `PYTHONPATH` picks
the engine worktree: the spec worktree checked out at `c3ee66f` for 2048,
the main one at `e863778` for 512), config = `path:` to the HF snapshot,
`reasoning_effort: low`, `kv_cache_bits: 16`, `speculative_decoding: true`
(the 96 GB defaults), port 1299; then `lmk bench --url … --seed 20261101`;
`lmk status` captured; stopped by PID. The resident lmk stays up, idle.

## Expectation

- prefill (4k cold probe): within 3% between arms (exp02: within 1% at
  4k–50k; the owner's row 314 vs the table's 321–323).
- cached prefill: the same (not chunk-dependent).
- decode prose / code: within ±5%, acceptance within ±5 points. Confidence:
  high — drafting and verification happen in decode steps, which exp02–05
  showed identical across chunks; the only prefill product the drafter uses is
  the final hidden state, which chunking does not change beyond float noise.

Reading rule: acceptance differing by > 10 points or decode by > 10% between
arms means the chunk reached the drafter and the design does not stand
until explained; otherwise the 512 row goes into `docs/benchmarks.md` and
the owner's 50% prose acceptance is the private variant's own number.

## Baseline

`docs/benchmarks.md` rows of 2026-09-24 (catalog 27B-4bit, kv8 + draft):
322 tok/s prefill, 45.0 prose / 60.1 code, 88% accepted — different KV bits
and engine; the 2048 arm here is the like-for-like.

## Run

```
research/2026-11-01-sink-mask-review/exp06-bench-with-draft-2048-vs-512/run.sh
```

## Attempt 1 (19:41): did not run

The temporary lmk exited at start with `LmkDraftUnusable`: the catalog
snapshot (lmstudio-community 4-bit) carries no MTP head of its own, and a
`path:` model does not get the model page's draft default, so
`speculative_decoding: true` needs `draft_weights: official` (lmk's copy of
the original head — the same head the owner's config uses). Added to
`run.sh`'s config; the failed serve log is `raw/attempt1-serve-2048-no-head.log`.
Attempt 2 is the measurement.

## Attempt 2 (19:46): did not run

Both temporary instances came up (20 s to ready, the official head reused
from `~/.lmk/drafts`), but `lmk bench --url …/v1` printed "no lmk answers":
`--url` takes the root (`http://host:port`), the command appends
`/lmk/v1/status` itself. Fixed in `run.sh`; logs kept as `raw/attempt2-*`.
Attempt 3 is the measurement.

## Result (attempt 3, 2026-11-01 19:48, `raw/bench-*.txt`, `raw/serve-*.log`)

| engine chunk | prefill (4,062 tok cold) | cached prefill | decode prose | decode code | accepted prose / code |
|---|---|---|---|---|---|
| 2048 (`c3ee66f`) | 324 tok/s | 53k tok/s, first token 0.94 s | 42.1 tok/s | 54.6 tok/s | 48% / 74% |
| 512 (`e863778`) | 314 tok/s | 53k tok/s, first token 0.94 s | 41.2 tok/s | 54.9 tok/s | 48% / 74% |

As expected: acceptance identical, decode within 2%, prefill −3% at 4k
(the same 314 the owner's run showed), cached prefill unchanged. The chunk
does not reach the drafter; the prose acceptance of ~50% with lmk's copy of
the official head is the head's own number on this prompt (thinking on,
effort low), also on the catalog model. Reading rule met: the 512 row goes
to `docs/benchmarks.md`.

Two notes on the raw rows: the "engine" column printed by `lmk bench` comes
from the lmk tree's `ENGINE_COMMIT` (both say `e863778`); the 2048 arm really
ran `c3ee66f` (`raw/run.log`). The model column shows the snapshot hash
because the temporary config used `path:`. "answers start prose ''" appears
in both arms and in the owner's run on the resident service — a pre-existing
display quirk of the bench with thinking on, not a chunk effect.
