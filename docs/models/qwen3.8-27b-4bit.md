# qwen3.8-27b-4bit

**The default.** Qwen3.8-27B, a dense 27B model with vision, quantized to 4 bits by LM Studio.
Text and images in, tool calls, thinking. This is the model lmk itself was built and measured against.

## Fits

| | |
|---|---|
| weights in memory | 16 GB |
| memory when loaded | 15 GiB (measured on the M3 Ultra) |
| needs at least (expected, not tested) | 32 GB — about 85k of context there; the full 262k from 64 GB up |
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
  reasoning_effort: low       # its template knows low / medium / xhigh; the default (xhigh) scored worse on every test
```

## Thinking

Thinking is on by default and at the highest level. `reasoning_effort` is not a sampling knob: the
template turns it into a system line at the very start of the prompt. `xhigh` (the default) says
"think carefully, validate key assumptions, consider plausible alternatives"; `low` says "keep your
thinking brief, move directly to the conclusion"; `medium` adds nothing. Those three are the only
accepted values; anything else stops `lmk up` with the template's message.
Server-wide, never per request — the setting sits at the start of every prompt, so changing it
makes every cached conversation cold once.

## Known issues

- **The default thinking level (`xhigh`) is the worst setting we measured.** On 50 GSM8K, 40 HumanEval,
  10 format and 5 tool tasks, 3 runs each, `xhigh` scored below `low` on every category and ran out
  of 8,000 tokens 20 times, thinking; `low` was the best configuration of any model in the test
  (code 99%, format 100%) at about three seconds more per answer. Hence the recommended config above.
- **With thinking off it guesses on small arithmetic:** asked whether 91 is prime it said yes 3 times
  in 3; with `low` thinking it got it right every time.
- **Thinks at the top level unless told otherwise.** The template's default effort is `xhigh`: a
  system line asking the model to "think carefully, validate key assumptions, consider plausible
  alternatives" goes in front of every prompt. On one agent task it thought through a 128-item
  list by hand (15,698 tokens in one step). `reasoning_effort: low` or `medium` if that bites.
- **Slower on long conversations:** ~33 tok/s at 60k+ tokens of context against ~39 fresh.
- **First touch after hours idle is slow.** If other models were loaded meanwhile, the weights get
  paged back in on the next request: 37 s instead of 13 s for a 4k prompt, once.

## Tested

- Integration tests: tool calls, cache hits, warm-up, image reading, cache surviving a restart — pass.
- Runs the author's agents daily (56-tool system prompt, conversations to 70k tokens).

- [intelligence eval](../../research/2026-09-23-intelligence-27b-vs-122b/README.md) (2026-09-23): thinking off / low / xhigh, 3 runs each — math 95 / 94 / 91%,
  code 96 / 99 / 89%, format 77 / 100 / 97%, tools 100% all three.

## Not tested

- Other Macs than the M3 Ultra 96 GB.
- `reasoning_effort` levels: wired through, not yet measured for speed or quality.
