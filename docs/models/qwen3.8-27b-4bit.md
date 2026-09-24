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
  kv_cache_bits: 8            # halves what each token of context costs: 122k tokens on a 32 GB Mac instead of 85k.
                              # Scored the same as 16-bit with 60k tokens of context in front of every task (below)
```

Speculative decoding (`speculative_decoding: true`, the model's own draft head that `lmk pull` fetches) is
measured but not recommended yet: see Tested.

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
- **`kv_cache_bits: 8` changes answers slightly.** The KV cache holds numbers rounded to 8 bits, so a greedy answer can
  differ from the 16-bit one after a few dozen tokens even on a 3k-token prompt. It did not score lower on our
  tests (Tested), and the prompt cache written at 8 bits lives in its own directory: switching bits starts the
  cache empty for this model. Restoring a cached prompt is about a fifth slower at 8 bits (first token 1.03 s
  instead of 1.01 s on a 4k prompt).
- **`speculative_decoding: true` helps one request at a time.** While two or more requests are being answered
  together lmk decodes them plainly (a mixed-length batch did not reproduce plain decoding on the engine's
  batched path; recorded in `research/2026-09-23-speculative-decoding/exp03-engine-wiring/`). Prose answers can
  differ from the plain ones (near-tie words flip); code and copy-editing answers came out token for token the
  same. With the model's default sampling the gain is smaller than with `temperature: 0`.


## Tested

- Integration tests: tool calls, cache hits, warm-up, image reading, cache surviving a restart — pass.
- Runs the author's agents daily (56-tool system prompt, conversations to 70k tokens).

- [intelligence eval](../../research/2026-09-23-intelligence-27b-vs-122b/README.md) (2026-09-23): thinking off / low / xhigh, 3 runs each — math 95 / 94 / 91%,
  code 96 / 99 / 89%, format 77 / 100 / 97%, tools 100% all three.
- **`kv_cache_bits: 8`, 60,000 tokens of project files in front of every task** (2026-09-23,
  `research/2026-09-23-kv-cache-quant/exp02-long-context-eval/`): code 40/40, instruct 20/20, tools 10/10 —
  16-bit scored 39/40, 20/20, 10/10; 4-bit 38/40, 19/20, 9/10. The engine's probe measured 34,816 bytes of KV per
  token at 8 bits (65,536 at 16). Chat tests with thinking on and off, and prompt-cache restore after a restart, pass at 8 bits.
- **`speculative_decoding: true`** (2026-09-23, `research/2026-09-23-speculative-decoding/exp03-engine-wiring/`):
  greedy decoding 39 → 58 tok/s on code and copy-editing (output identical), 39 → 46 tok/s on a story (output differs);
  with the model's default sampling 39 → 50 tok/s on code (69% of drafted tokens accepted). Chat tests with thinking
  on and off pass with the draft loaded. Whether answers under default sampling score the same is being measured.


## Not tested

- Other Macs than the M3 Ultra 96 GB.
- `reasoning_effort` levels: wired through, not yet measured for speed or quality.
