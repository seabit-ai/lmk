# qwen3.5-122b-a10b-4bit

Qwen3.5-122B-A10B: a mixture-of-experts model, 122B parameters of which 10B are active per token,
256 experts, with vision. Quantized to 4 bits by mlx-community. Text and images in, tool calls.

**Read the Thinking section before using it for an agent.**

## Fits

| | |
|---|---|
| weights in memory | 70 GB |
| memory when loaded | 65 GiB (measured on the M3 Ultra) |
| needs at least (expected, not tested) | 96 GB — about 167k of context there; the full 262k from 128 GB up |
| context on a 96 GB Mac | 165,888 — lowered from 262,144 to fit; about 10 GB is left for conversations |
| download | 69.6 GB, `lmk pull` |

It needs the GPU to itself: nothing else that holds GPU memory can run beside it on a 96 GB Mac.

## Speed (M3 Ultra, 96 GB — [`docs/benchmarks.md`](../benchmarks.md))

| prefill | cached prefill | decode |
|---|---|---|
| 753 tok/s | 89k tok/s | 60 tok/s |

Faster than the 27B on every count: prefill compute and decode bytes both scale with the 10B
active parameters, and its cache is smaller per token (24 KB against 64 KB).

## Recommended configuration

```yaml
model:
  name: qwen3.5-122b-a10b-4bit
  thinking: false
```

## Thinking

**Off is the usable setting for an agent.** With thinking on, this model can think for thousands of
tokens on a small task and never answer: any count in the request ("a 250-word story", "about 250
words", "exactly three bullets") sends it into counting the words of its draft one by one — 6,000
tokens of thinking and no story, where the 27B answers in 3,500. Its template has only an on/off
switch, no effort levels. With `thinking: false` it answered the same story in 402 tokens and
called tools correctly in a few dozen tokens per step.

Server-wide, never per request — the setting sits at the start of every prompt, so changing it
makes every cached conversation cold once.

## Known issues

- **Not smarter than the 27B on our tests — faster and terser.** With thinking off on both ([intelligence eval](../../research/2026-09-23-intelligence-27b-vs-122b/README.md)):
  math 95% each, code 91% against the 27B's 96%, format 83% against 77%, tools 100% both; it answered
  in 6 s / 218 tokens median against 9 s / 255. It has no thinking level between off and on, so the
  27B's best setting (`reasoning_effort: low`, code 99%, format 100%) has no counterpart here.
- **With thinking on, any count in the request can send it into a loop:** "a 250-word story",
  "about 250 words", "exactly three bullets" — it drafts, then counts the words of its draft one by
  one, and runs out of tokens without answering (6,000 tokens, no story). `thinking: false` avoids it.
- **The context is cut to 165k on a 96 GB Mac**, and nothing else can hold GPU memory beside it.
- **Two long conversations at once fill it up:** the token budget for parallel requests is about
  430k against 980k for the 27B.

## Tested

- Integration tests with thinking on and with thinking off: tool calls, cache hits, warm-up, image reading,
  cache surviving a restart — pass both ways.
- With thinking off, three agent-shaped tasks: a two-step read-then-edit (the edit quoted the file
  content exactly), a question that needs no tool (none called), an integer argument typed as an
  integer.

- [intelligence eval](../../research/2026-09-23-intelligence-27b-vs-122b/README.md) (2026-09-23), thinking off, 3 runs each: math 143/150, code 109/120, format 25/30, tools 15/15.

## Not tested

- Quality against the 27B on hard, long agent work; the eval's tasks are small and the 27B ties or wins them.
- Thinking on, at all: every listed number is thinking off.
- Long conversations and many-tool prompts with thinking off.
- Any Mac other than the M3 Ultra 96 GB; on less than 96 GB it does not load.
