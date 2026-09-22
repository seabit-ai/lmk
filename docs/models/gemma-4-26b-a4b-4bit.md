# gemma-4-26b-a4b-4bit

Google's Gemma 4 26B-A4B: a mixture of experts, 26B parameters of which 4B are active per token,
with vision, quantized to 4 bits by mlx-community. Text and images in, tool calls. The first model
in lmk that is not a Qwen — its chat template, thinking markers and tool-call format are its own,
and lmk handles them.

The fastest model here, and the smallest of the tested ones: 16 GB of weights.

## Fits

| | |
|---|---|
| weights in memory | 16 GB |
| memory when loaded | 14 GiB (measured on the M3 Ultra) |
| needs at least (expected, not tested) | 32 GB — leaves about 9 GiB for conversations |
| context on a 96 GB Mac | 262,144 (its maximum) |
| download | 15.6 GB, `lmk pull` |

## Speed (M3 Ultra, 96 GB — [`docs/benchmarks.md`](../benchmarks.md))

| prefill | cached prefill | decode |
|---|---|---|
| 1,833 tok/s | 83k tok/s | 120 tok/s |

Three times the decode speed of the Qwen3.8-27B and five to six times its prompt reading: 4B active
parameters instead of 27B.

## Recommended configuration

```yaml
model:
  name: gemma-4-26b-a4b-4bit
```

## Thinking

**Off by default**, the opposite of the Qwen models. `thinking: true` asks for it; the model then
decides per turn whether to think — an obvious next step such as "read this file" gets a tool call
with no thinking at all. And off is a hint, not a lock: on a one-sentence question it still thought
for about 250 tokens before answering, and once before a plain file read. Either way lmk splits the thinking out into `reasoning_content`.
Its template has no effort levels. Server-wide, never per request — the setting sits at the start
of every prompt, so changing it makes every cached conversation cold once.

## Tested

- Integration tests with thinking off and on: tool calls, cache hits, warm-up, image reading, cache
  surviving a restart — pass both ways.
- Three agent-shaped tasks (read-then-edit, a no-tool question, an integer argument): all correct,
  21–43 tokens per tool step.

## Not tested

- Quality against the Qwen models on real agent work.
- Long conversations and many-tool prompts.
- Any Mac other than the M3 Ultra 96 GB.
