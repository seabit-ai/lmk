# qwen3.8-27b-4bit

**The default.** Qwen3.8-27B, a dense 27B model with vision, quantized to 4 bits by LM Studio.
Text and images in, tool calls, thinking. This is the model lmk itself was built and measured against.

## Fits

| | |
|---|---|
| weights in memory | 16 GB |
| context on a 96 GB Mac | 262,144 (its maximum) |
| download | 16.1 GB, `lmk pull` |

## Speed (M3 Ultra, 96 GB — [`docs/benchmarks.md`](../benchmarks.md))

| prefill | cached prefill | decode |
|---|---|---|
| 323 tok/s | 53k tok/s | 39 tok/s (about 33 on long agent conversations) |

## Recommended configuration

```yaml
model:
  name: qwen3.8-27b-4bit
  # reasoning_effort: low     # this model's template knows low / medium / high; default is the highest
```

## Thinking

Thinking is on by default and at the highest level: the template has no default effort, so every
request runs at the top tier. Lower it with `reasoning_effort` if answers spend too long thinking.
Server-wide, never per request — the setting sits at the start of every prompt, so changing it
makes every cached conversation cold once.

## Tested

- Integration tests: tool calls, cache hits, warm-up, image reading, cache surviving a restart — pass.
- Runs the author's agents daily (56-tool system prompt, conversations to 70k tokens).

## Not tested

- Other Macs than the M3 Ultra 96 GB.
- `reasoning_effort` levels: wired through, not yet measured for speed or quality.
