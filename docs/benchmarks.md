# Benchmarks

What `lmk bench` measures, from outside the service over HTTP, one request at a time, `temperature: 0`:

- **cold prefill** — a fixed ~4k-token text behind a random number, so nothing in the cache matches: tokens/s of reading a prompt lmk has never seen.
- **cached prefill** — the same request again: tokens/s at which the cached part comes back from disk (the server's own restore timing). The first-token time next to it is what a client waits: the cache restores to the last 256-token boundary, so up to 255 tokens are still computed, plus HTTP.
- **decode** — a short prompt, 400 tokens out: tokens/s while writing. Measured twice, on prose (a story) and on code (a class):
  speculative decoding gains most on code, so one prose number would hide it. Where a draft model was in use, the share of
  drafted tokens the model accepted is next to the code number.

The model cell names the switches a row was measured with when they are not the defaults (`KV cache 8-bit`, `speculative decoding`);
`lmk bench` prints them at the top too, so a row is never a mystery number.

A warm-up request runs first and is not counted: after hours idle, or after other models were loaded, the first touch of the weights pages them back in (37 s instead of 12.8 s for the cold probe, once, on the M3 Ultra below).

Run it on your Mac and paste the row into an [issue](https://github.com/seabit-ai/lmk/issues) — different machines and models are what this table is missing:

```sh
lmk bench
```

Each run prints the seed behind its cold prompt. `lmk bench --seed <that number>` later — after
`lmk down && lmk up`, or after a reboot — sends the same prompt again: if it comes back from the
cache, the cache survived, and the run says so instead of reporting a cold number.

| chip | memory | model | context | prefill | cached prefill | decode | lmk | engine | date |
|---|---|---|---|---|---|---|---|---|---|
| Apple M3 Ultra | 96 GB | qwen3.8-27b-4bit | 262,144 | 323 tok/s (4,074 tokens) | 53k tok/s (3,840 cached; first token 1.02 s) | 39.5 tok/s | v0.2.0+ | 08f0c07 | 2026-09-22 |
| Apple M3 Ultra | 96 GB | qwen3.8-27b-8bit | 262,144 | 319 tok/s (4,074 tokens) | 44k tok/s (3,840 cached; first token 1.07 s) | 23.1 tok/s | v0.2.0+ | 08f0c07 | 2026-09-22 |
| Apple M3 Ultra | 96 GB | qwen3.8-27b-5bit | 262,144 | 315 tok/s (4,074 tokens) | 57k tok/s (3,840 cached; first token 1.04 s) | 31.6 tok/s | v0.6.0+ | 08f0c07 | 2026-09-22 |
| Apple M3 Ultra | 96 GB | qwen3.8-27b-6bit | 262,144 | 315 tok/s (4,074 tokens) | 53k tok/s (3,840 cached; first token 1.06 s) | 28.1 tok/s | v0.6.0+ | 08f0c07 | 2026-09-22 |
| Apple M3 Ultra | 96 GB | qwen3.8-27b-4bit | 262,144 | 324 tok/s (4,074 tokens) | 58k tok/s (3,840 cached; first token 1.01 s) | 39.5 tok/s | dev | d3650db | 2026-09-23 |
| Apple M3 Ultra | 96 GB | qwen3.8-27b-4bit (KV cache 8-bit) | 262,144 | 323 tok/s (4,074 tokens) | 47k tok/s (3,840 cached; first token 1.03 s) | 38.9 tok/s | dev | d3650db | 2026-09-23 |
| Apple M3 Ultra | 96 GB | qwen3.8-27b-4bit (speculative decoding) | 262,144 | 323 tok/s (4,062 tokens) | 56k tok/s (3,840 cached; first token 0.92 s) | 40.5 tok/s prose (greedy code: 58) | dev | 2839cfa | 2026-09-23 |
| Apple M3 Ultra | 96 GB | qwen3.8-27b-4bit (KV cache 8-bit, speculative decoding) — engine on mlx-vlm 0.6.12 | 262,144 | 321 tok/s (4,034 tokens) | 47k tok/s (3,840 cached; first token 0.91 s) | 44.5 tok/s prose · 58.6 code (88% of drafted tokens accepted) | dev | 7a1e17f | 2026-09-24 |
| Apple M3 Ultra | 96 GB | qwen3.8-27b-4bit (KV cache 8-bit, speculative decoding) — engine on mlx-vlm 0.6.16 with its exact verifier | 262,144 | 322 tok/s (4,034 tokens) | 48k tok/s (3,840 cached; first token 0.91 s) | 37.3 tok/s prose · 49.6 code (88% of drafted tokens accepted) | dev | bc9588e | 2026-09-24 |
| Apple M3 Ultra | 96 GB | qwen3.8-27b-4bit (KV cache 8-bit, speculative decoding) — the model page's recommendation | 262,144 | 322 tok/s (4,034 tokens) | 49k tok/s (3,840 cached; first token 0.91 s) | 45.0 tok/s prose · 60.1 code (88% of drafted tokens accepted) | v0.7.1+ | 25d1c38 | 2026-09-24 |
| Apple M3 Ultra | 96 GB | qwen3.5-122b-a10b-4bit | 165,888 (lowered from 262,144 to fit) | 753 tok/s (4,032 tokens) | 89k tok/s (3,840 cached; first token 0.50 s) | 60.5 tok/s | v0.3.0+ | 08f0c07 | 2026-09-22 |
| Apple M3 Ultra | 96 GB | qwen3.5-122b-a10b-48gb | 262,144 | 746 tok/s (4,034 tokens) | 89k tok/s (3,840 cached; first token 0.53 s) | 53.7 tok/s | v0.4.0+ | 08f0c07 | 2026-09-22 |
| Apple M3 Ultra | 96 GB | gemma-4-26b-a4b-4bit | 262,144 | 1,833 tok/s (4,028 tokens) | 83k tok/s (3,840 cached; first token 0.26 s) | 119.5 tok/s | v0.4.1+ | 08f0c07 | 2026-09-22 |
| Apple M3 Ultra | 96 GB | gemma-4-e4b-4bit | 131,072 | 2,199 tok/s (4,028 tokens) | 183k tok/s (3,840 cached; first token 0.19 s) | 93.5 tok/s | v0.5.0+ | 08f0c07 | 2026-09-22 |
| Apple M3 Ultra | 96 GB | gemma-4-31b-4bit | 262,144 | 252 tok/s (4,028 tokens) | 26k tok/s (3,840 cached; first token 1.23 s) | 33.0 tok/s | v0.6.0+ | 08f0c07 | 2026-09-22 |
| Apple M3 Ultra | 96 GB | gemma-4-12b-4bit ([tried, not listed](models/gemma-4-12b-4bit.md)) | 262,144 | 666 tok/s (4,028 tokens) | 52k tok/s (3,840 cached; first token 0.51 s) | 71.8 tok/s | v0.5.0+ | 08f0c07 | 2026-09-22 |

Numbers from real agent traffic (a 56-tool, 11k-token system prompt; a 27k-token conversation) are in the [README](../README.md); they differ from these synthetic probes and both are true under their conditions.
