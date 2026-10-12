`create_generator`'s docstring promises `seed (Optional[int]): Random seed for reproducible generation`, but the batched path drops it — the code says so itself:

```python
seed: Optional[int] = None,  # Seed arg is ignored for batched gen
```

(`generate.py`, `_batched_generation`; unchanged on current `main`.)

The sequential path reseeds the process-global RNGs per request (`set_seed`), so concurrent requests disturb each other, and a replay only matches if the entire interleaving matches.

We run per-request seeds instead: each request carries a uint64 seed, and every sampling draw is keyed by *(seed, position of the token being generated)*. A replay then matches regardless of batch mates, restarts, or how much prefix came back from the prompt cache; temp-0 takes no draws; nothing global is reseeded. In production ~3 weeks — we use it to replay prod requests when debugging (same seed, same request → same distribution; divergence then points at real state).

Code, MIT, on our fork: [`bab3536`](https://github.com/seabit-ai/mlx-engine/commit/bab353600efddd9abe762ae0838194ef4dae59e3) (batched path), [`c2c6f9e`](https://github.com/seabit-ai/mlx-engine/commit/c2c6f9ecbada08263c6e16758b3962a1183e8184) (every path, uint64), [tests](https://github.com/seabit-ai/mlx-engine/blob/lmk/tests/test_seeded_sampling.py) (33). They sit on a branch with our other changes, so they won't cherry-pick cleanly — if there's interest I'll rebase onto `main` as a standalone PR. Opening this issue first per CONTRIBUTING.
