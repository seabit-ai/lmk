`create_generator`'s docstring promises `seed (Optional[int]): Random seed for reproducible generation`, but the batched path drops it — the code says so itself:

```python
seed: Optional[int] = None,  # Seed arg is ignored for batched gen
```

(`generate.py`, `_batched_generation` — present at `08f0c07` and unchanged on current `main`.)

The sequential path handles it by calling `set_seed(seed)` per request, which reseeds the process-global RNGs (mlx, numpy, torch, `random`). That is reproducible for one request at a time, but two concurrent requests reseed each other mid-generation, and a replay only matches if the entire interleaving matches.

**What we run instead** (in production for ~3 weeks behind <https://github.com/seabit-ai/lmk>): each request carries its own uint64 seed, and every sampling draw is keyed by *(seed, position of the token being generated)*. Properties that fall out:

- a replay with the same seed matches regardless of what else shares the batch, of process restarts, or of how much prefix came back from the prompt cache;
- greedy requests take no draws, so temp-0 behaviour is untouched;
- no global reseeding anywhere on the request path, so concurrent requests cannot disturb each other.

We lean on it daily to replay production requests when debugging (same seed, same request → same distribution over tokens; divergence then points at real state, e.g. it is how we found a cache-restore positioning bug on our side).

Code, MIT, on our fork:

- [`bab3536`](https://github.com/seabit-ai/mlx-engine/commit/bab353600efddd9abe762ae0838194ef4dae59e3) — batched path honours a per-request seed, each draw keyed by (seed, generated position)
- [`c2c6f9e`](https://github.com/seabit-ai/mlx-engine/commit/c2c6f9ecbada08263c6e16758b3962a1183e8184) — seeds on every path, as uint64
- tests: [`tests/test_seeded_sampling.py`](https://github.com/seabit-ai/mlx-engine/blob/lmk/tests/test_seeded_sampling.py)

The commits sit on a branch that carries our other changes, so they won't cherry-pick cleanly as-is. If there's interest, I'll rebase them onto current `main` as a standalone PR — opening this issue first since CONTRIBUTING.md asks for discussion before feature PRs.
