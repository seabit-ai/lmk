# gemma-4-e4b-4bit

Google's Gemma 4 E4B ("effective 4B"), the small one, with vision, quantized to 4 bits by LM Studio.
Text and images in, tool calls. 7 GB of weights: the model for a 16 GB Mac.

## Fits

| | |
|---|---|
| weights in memory | 7 GB |
| memory when loaded | 6 GiB (measured on the M3 Ultra) |
| needs at least (expected, not tested) | 16 GB — about 75k of context there; the full 131k from 24 GB up |
| context on a 96 GB Mac | 131,072 (its maximum) |
| download | 6.8 GB, `lmk pull` |

## Speed (M3 Ultra, 96 GB — [`docs/benchmarks.md`](../benchmarks.md))

| prefill | cached prefill | decode |
|---|---|---|
| 2,199 tok/s | 183k tok/s | 94 tok/s |

Reads prompts faster than anything else here; writes a little slower than the 26B-A4B mixture of
experts (94 against 120 tok/s) because all of its 4B are active for every token.

## Recommended configuration

```yaml
model:
  name: gemma-4-e4b-4bit
```

## Thinking

Off by default, like the other Gemma 4 models; `thinking: true` lets the model decide per turn.
Off is a hint, not a lock. No effort levels. Server-wide, never per request.

## Known issues

- **Wordy.** A tool step that the 26B-A4B does in 21–43 tokens takes this model 70–160: it writes a
  sentence or two around the call. Correct, but slower than the token rate suggests.
- **Thinks with thinking off.** On a one-sentence question it thought for ~250 tokens first.
- **131k context**, half the other models'.

## Tested

- Integration tests with thinking off and on: tool calls, cache hits, warm-up, image reading, cache
  surviving a restart — pass both ways.
- Three agent-shaped tasks (read-then-edit, a no-tool question, an integer argument), four runs
  (one at temperature 0, three with the model's own sampling): all correct every time.

## Not tested

- Quality on anything harder than those tasks; a 4B model has limits the tasks did not reach.
- A 16 GB or 24 GB Mac.
