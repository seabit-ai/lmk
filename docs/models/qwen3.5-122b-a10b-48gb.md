# qwen3.5-122b-a10b-48gb

The same Qwen3.5-122B-A10B mixture of experts as [qwen3.5-122b-a10b-4bit](qwen3.5-122b-a10b-4bit.md),
quantized by a community author (baa-ai) to fit in less memory: the 256 routed experts are at 2 and 3 bits,
attention and the shared expert stay at 5–8 bits. Text and images in, tool calls.

This is what to try if the 4-bit does not fit your Mac. It is not from the model's authors or from
LM Studio / mlx-community, and 2-bit experts are an aggressive setting — read Not tested.

**Read the Thinking section before using it for an agent.**

## Fits

| | |
|---|---|
| weights in memory | 44 GB |
| memory when loaded | 44 GiB (measured on the M3 Ultra) |
| needs at least (expected, not tested) | 64 GB — leaves about 5 GiB for conversations |
| context on a 96 GB Mac | 262,144 (its maximum), with room for two long conversations at once |
| download | 47.2 GB, `lmk pull` |

## Speed (M3 Ultra, 96 GB — [`docs/benchmarks.md`](../benchmarks.md))

| prefill | cached prefill | decode |
|---|---|---|
| 746 tok/s | 89k tok/s | 54 tok/s |

Slower to write than the 4-bit (60 tok/s), although it reads fewer bytes per token: what you get for
the smaller size is memory, not speed.

## Recommended configuration

```yaml
model:
  name: qwen3.5-122b-a10b-48gb
  thinking: false
```

## Thinking

Same as the 4-bit: **off is the usable setting for an agent.** With thinking on, any count in the
request sends this model into counting the words of its draft one by one until it runs out of
tokens. Its template has only an on/off switch. Server-wide, never per request — changing it makes
every cached conversation cold once.

## Tested

- Integration tests with thinking on and off: tool calls, cache hits, warm-up, image reading, cache
  surviving a restart — pass both ways.
- With thinking off, three agent-shaped tasks (read-then-edit, a no-tool question, an integer
  argument): all correct, and the answers were nearly word for word the 4-bit's.

## Not tested

- Quality on hard tasks. 2-bit experts lose more than 4-bit ones; on our four small tasks the loss did
  not show, which is not evidence that it never will.
- A 64 GB Mac, or any Mac other than the M3 Ultra 96 GB.
- Long conversations and many-tool prompts.
