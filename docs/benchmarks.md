# Benchmarks

What `lmk bench` measures, from outside the service over HTTP, one request at a time, `temperature: 0`:

- **cold prefill** — a fixed ~4k-token text behind a random number, so nothing in the cache matches: tokens/s of reading a prompt lmk has never seen.
- **cache-hit first token** — the same request again: seconds until the first token when the prompt comes back from disk (the cache restores to the last 256-token boundary, so a few tokens are still computed).
- **decode** — a short prompt, 400 tokens out: tokens/s while writing.

A warm-up request runs first and is not counted: after hours idle, or after other models were loaded, the first touch of the weights pages them back in (37 s instead of 12.8 s for the cold probe, once, on the M3 Ultra below).

Run it on your Mac and paste the row into an [issue](https://github.com/seabit-ai/lmk/issues) — different machines and models are what this table is missing:

```sh
lmk bench
```

Each run prints the seed behind its cold prompt. `lmk bench --seed <that number>` later — after
`lmk down && lmk up`, or after a reboot — sends the same prompt again: if it comes back from the
cache, the cache survived, and the run says so instead of reporting a cold number.

| chip | memory | model | context | cold prefill | cache-hit first token | decode | lmk | engine | date |
|---|---|---|---|---|---|---|---|---|---|
| Apple M3 Ultra | 96 GB | qwen3.8-27b-4bit | 262,144 | 323 tok/s (4,074 tokens) | 1.02 s (3,840 cached) | 39.5 tok/s | v0.1.0+ | 08f0c07 | 2026-09-22 |
| Apple M3 Ultra | 96 GB | qwen3.8-27b-8bit | 262,144 | 319 tok/s (4,074 tokens) | 1.07 s (3,840 cached) | 23.1 tok/s | v0.1.0+ | 08f0c07 | 2026-09-22 |

Numbers from real agent traffic (a 56-tool, 11k-token system prompt; a 27k-token conversation) are in the [README](../README.md); they differ from these synthetic probes and both are true under their conditions.
