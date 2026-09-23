# qwen3.8-27b-6bit

The same Qwen3.8-27B as the default, quantized to 6 bits by LM Studio: the step below the 8-bit at
three quarters of its size. Text and images in, tool calls, thinking.

## Fits

| | |
|---|---|
| weights in memory | 23 GB |
| memory when loaded | 21 GiB (measured on the M3 Ultra) |
| needs at least (expected, not tested) | 36 GB — about 53k of context there; the full 262k from 64 GB up |
| context on a 96 GB Mac | 262,144 (its maximum) |
| download | 22.8 GB, `lmk pull` |

On a 32 GB Mac it loads but leaves room for only ~18k of context, too little for an agent; on 48 GB
it leaves 157k where the 8-bit leaves 90k.

## Speed (M3 Ultra, 96 GB — [`docs/benchmarks.md`](../benchmarks.md))

| prefill | cached prefill | decode |
|---|---|---|
| 315 tok/s | 53k tok/s | 28 tok/s |

Reads prompts as fast as the 4-bit; writes 30% slower (28 against 39.5) and 20% faster than the 8-bit (23).

## Recommended configuration

```yaml
model:
  name: qwen3.8-27b-6bit
  # reasoning_effort: low     # this model's template knows low / medium / xhigh; default is xhigh, the highest
```

## Thinking

Same as [qwen3.8-27b-4bit](qwen3.8-27b-4bit.md): on by default at the highest level, `reasoning_effort` to lower it, server-wide.

## Known issues

- Same as the 4-bit: top-level thinking by default, slower on long conversations, a slow first touch
  after idle.

## Tested

- Integration tests with thinking on and off: tool calls, cache hits, warm-up, image reading, cache
  surviving a restart — pass both ways.
- Three agent-shaped tasks (read-then-edit, a no-tool question, an integer argument), three runs
  with the model's own sampling: all correct every time.

## Not tested

- Whether 6-bit answers are measurably better than 4-bit or 5-bit on real work; we have no quality benchmark.
- Any Mac other than the M3 Ultra 96 GB.
