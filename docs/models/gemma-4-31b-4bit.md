# gemma-4-31b-4bit

Google's Gemma 4 31B, a dense model with vision, quantized to 4 bits by LM Studio. Text and images
in, tool calls. The other family at the size of the default Qwen3.8-27B: pick it if you want
Gemma's behaviour — no thinking unless asked, answers straight away — at that size.

## Fits

| | |
|---|---|
| weights in memory | 18 GB |
| memory when loaded | 17 GiB (measured on the M3 Ultra) |
| needs at least (expected, not tested) | 32 GB — about 39k of context there; the full 262k from 96 GB up |
| context on a 96 GB Mac | 262,144 (its maximum) |
| download | 18.4 GB, `lmk pull` |

Its cache is the largest per token of the listed models (80 KB against the Qwen 27B's 64 KB), so the
same memory holds a shorter conversation: about 150k on a 48 GB Mac where the Qwen fits 224k.

## Speed (M3 Ultra, 96 GB — [`docs/benchmarks.md`](../benchmarks.md))

| prefill | cached prefill | decode |
|---|---|---|
| 252 tok/s | 26k tok/s | 33 tok/s |

Slower than the Qwen3.8-27B-4bit on all three (323 / 53k / 39.5): more weight bytes per token and a
bigger cache to read back.

## Recommended configuration

```yaml
model:
  name: gemma-4-31b-4bit
```

## Thinking

Off by default, like the other Gemma 4 models; `thinking: true` lets the model decide per turn.
Off is a hint, not a lock. No effort levels. Server-wide, never per request.

## Known issues

- **Slower than the Qwen at the same size**, and its cache takes twice as long to come back from disk
  (a 27k conversation resumes in about 1 s against 0.4 s).
- **Shorter context per GB of memory** than any other listed model.
- Thinks briefly with thinking off, as the other Gemma 4 models do.

## Tested

- Integration tests with thinking off and on: tool calls, cache hits, warm-up, image reading, cache
  surviving a restart — pass both ways.
- Three agent-shaped tasks (read-then-edit, a no-tool question, an integer argument), three runs
  with the model's own sampling: all correct every time (60–100 tokens per tool step).

## Not tested

- Quality against the Qwen3.8-27B on real agent work: the tasks here are small, and the two families
  may differ most where the tasks are hard.
- Any Mac other than the M3 Ultra 96 GB.
