# qwen3.8-27b-5bit

The same Qwen3.8-27B as the default, quantized to 5 bits by LM Studio: a step between the 4-bit
(16 GB) and the 8-bit (30 GB). Text and images in, tool calls, thinking.

## Fits

| | |
|---|---|
| weights in memory | 19 GB |
| memory when loaded | 18 GiB (measured on the M3 Ultra) |
| needs at least (expected, not tested) | 32 GB — about 52k of context there; the full 262k from 64 GB up |
| context on a 96 GB Mac | 262,144 (its maximum) |
| download | 19.4 GB, `lmk pull` |

On a 32 GB Mac the 4-bit leaves room for 85k of context; this one for 52k. That is the trade.

## Speed (M3 Ultra, 96 GB — [`docs/benchmarks.md`](../benchmarks.md))

| prefill | cached prefill | decode |
|---|---|---|
| 315 tok/s | 57k tok/s | 32 tok/s |

Reads prompts as fast as the 4-bit; writes 20% slower (32 against 39.5), as the extra weight bytes
predict.

## Recommended configuration

```yaml
model:
  name: qwen3.8-27b-5bit
  # reasoning_effort: low     # this model's template knows low / medium / xhigh; default is xhigh, the highest
```

## Thinking

Same as [qwen3.8-27b-4bit](qwen3.8-27b-4bit.md): on by default at the highest level, `reasoning_effort` to lower it, server-wide.

## Known issues

- **Misread a digit in our image test with thinking on** (4217 read as 4917, repeatable at
  temperature 0); with thinking off it read it right. The 4-bit reads it right both ways. One image,
  one digit — a data point, not a verdict on its vision.
- Same as the 4-bit otherwise: top-level thinking by default, slower on long conversations, a slow
  first touch after idle.

## Tested

- Integration tests: thinking off — all pass; thinking on — tool calls, cache hits, warm-up and cache
  across a restart pass, the image test misreads one digit (above).
- Three agent-shaped tasks (read-then-edit, a no-tool question, an integer argument), three runs
  with the model's own sampling: all correct every time.

## Not tested

- Whether 5-bit answers are measurably better than 4-bit on real work; we have no quality benchmark.
- Any Mac other than the M3 Ultra 96 GB.
