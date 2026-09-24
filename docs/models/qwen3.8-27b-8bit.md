# qwen3.8-27b-8bit

The same Qwen3.8-27B as the default, quantized to 8 bits: less quantization loss, twice the
weight bytes. Text and images in, tool calls, thinking.

## Fits

| | |
|---|---|
| weights in memory | 30 GB |
| memory when loaded | 27 GiB (measured on the M3 Ultra) |
| needs at least (expected, not tested) | 48 GB — about 89k of context there; the full 262k from 96 GB up |
| context on a 96 GB Mac | 262,144 (its maximum) |
| download | 29.5 GB, `lmk pull` |

## Speed (M3 Ultra, 96 GB — [`docs/benchmarks.md`](../benchmarks.md))

| prefill | cached prefill | decode |
|---|---|---|
| 319 tok/s | 44k tok/s | 23 tok/s |

Reading prompts is as fast as 4-bit (compute-bound); writing is 40% slower (memory-bandwidth-bound:
every token reads all the weights once). For an agent, most time goes to reading and cache hits,
so it feels closer to 4-bit than the decode number suggests.

## Recommended configuration

```yaml
model:
  name: qwen3.8-27b-8bit
  reasoning_effort: low       # its template knows low / medium / xhigh; the default (xhigh) scored worse on every test
```

## Thinking

Same as [qwen3.8-27b-4bit](qwen3.8-27b-4bit.md): on by default at the highest level, `reasoning_effort` to lower it, server-wide.

## Known issues

- **Use `reasoning_effort: low`**, as for the 4-bit: the default `xhigh` scored worse on every test and
  ran out of tokens thinking (measured on the 4-bit; the same template).
- Same as [qwen3.8-27b-4bit](qwen3.8-27b-4bit.md): top-level thinking by default, slower on long
  conversations, a slow first touch after idle.
- **Writes 40% slower than the 4-bit** (23 against 39 tok/s); reading prompts is as fast.

## Tested

- Integration tests: tool calls, cache hits, warm-up, image reading, cache surviving a restart — pass.

- Not run through the [intelligence eval](../../research/2026-09-23-intelligence-27b-vs-122b/README.md) itself; the 4-bit was, and the thinking findings carry over (same template).

## Not tested

- Whether the 8-bit quality difference is noticeable on agent tasks; we have not compared answers.
- Other Macs than the M3 Ultra 96 GB.
