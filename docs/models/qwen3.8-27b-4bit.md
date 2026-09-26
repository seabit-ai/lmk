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

| | prefill | cached prefill | decode, prose | decode, code |
|---|---|---|---|---|
| defaults (16-bit KV cache, no draft) | 323 tok/s | 53k tok/s | 39 tok/s (about 33 on long agent conversations) | — |
| the recommended configuration below | 322 tok/s | 49k tok/s | 45.0 tok/s | 60.1 tok/s (88% of drafted tokens accepted) |
| same, an agent request that carries `tools` | — | — | — | 58.2 tok/s against 36.5 without the draft (a tool call writing a file; 85% of drafted tokens in the call accepted) |

With the draft on, code and copy-editing come out 1.5x faster, prose 15% faster; answers scored the same as
plain decoding but are not token-for-token identical to it — at near-tie words (scores one floating-point step
apart) the checked batch and the one-token step can pick differently. On short prompts code and copy-editing
happened to match exactly; at 8k–128k context every greedy test diverged somewhere, with 8- and 16-bit KV cache. See "Tested" for how this was measured.

How it holds up as the conversation grows (the recommended configuration, thinking off, one request, the prefix
already cached; 256 tokens, greedy unless noted; [`research/2026-09-25-spec-long-context`](../../research/2026-09-25-spec-long-context/)):

| context | code, no draft → draft | prose, no draft → draft |
|---|---|---|
| 8k | 35 → 53 tok/s (1.50x) | 35 → 39 (1.12x) |
| 32k | 28 → 42 (1.51x) | 28 → 35 (1.27x) |
| 64k | 22 → 31 (1.43x) | 22 → 27 (1.24x) |
| 128k | 15 → 20 (1.28x) | 15 → 19 (1.23x) |
| 32k, sampled (temp 1.0) | 27 → 38 (1.40x) | 27 → 28 (1.02x) |

The draft never made decoding slower, up to 128k. Code gains less at long context because a verify step costs more
there, not because fewer drafts are accepted; sampled prose gains almost nothing.

Two drafts exist for this model; `model.draft` picks one, the default is `dflash2`:

| `draft:` | what it is | download | prose | code | copy-editing |
|---|---|---|---|---|---|
| `mtp` | the model's own multi-token-prediction head, split out of the original weights | 0.8 GB | 45 tok/s (1.2x) | 60 (1.5x) | 59 (1.5x) |
| `dflash2` (default) | z-lab's DFlash 2, a block-diffusion drafter trained for this model (Inco AI), quantized by us to 4 bits | 1.1 GB | 47 (1.2x) | 68 (1.7x) | 80 (2.1x) |

Measured at 16-bit KV cache, greedy; with `kv_cache_bits: 8` the numbers are the same within 2%. On the M3 Ultra
`dflash2` is faster on code and copy-editing and even on prose; the output is the same as with `mtp` (identical to
plain decoding on code and copy-editing). On a Mac with less memory bandwidth (M4 Pro, 48 GB) it should pull further
ahead — not measured: a `lmk bench` row from such a Mac would settle it.

## Recommended configuration

```yaml
model:
  name: qwen3.8-27b-4bit
  reasoning_effort: low       # its template knows low / medium / xhigh; the default (xhigh) scored worse on every test
  kv_cache_bits: 8            # halves what each token of context costs: 122k tokens on a 32 GB Mac instead of 85k.
                              # Scored the same as 16-bit with 60k tokens of context in front of every task (below)
  speculative_decoding: true  # the model's own draft head (lmk up fetches it): code 1.5x faster when one request is
                              # being answered; prose 1.15x; scored the same (below), not token-for-token identical
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
- **`kv_cache_bits: 8` changes answers slightly.** The KV cache holds numbers rounded to 8 bits, so a greedy answer can
  differ from the 16-bit one after a few dozen tokens even on a 3k-token prompt. It did not score lower on our
  tests (Tested), and the prompt cache written at 8 bits lives in its own directory: switching bits starts the
  cache empty for this model. Restoring a cached prompt is about a fifth slower at 8 bits (first token 1.03 s
  instead of 1.01 s on a 4k prompt).
- **`speculative_decoding: true` helps one request at a time.** While two or more requests are being answered
  together lmk decodes them plainly (a mixed-length batch did not reproduce plain decoding on the engine's
  batched path; recorded in `research/2026-09-23-speculative-decoding/exp03-engine-wiring/`). Answers can
  differ from the plain ones at near-tie words (on short prompts code and copy-editing matched exactly; at
  8k–128k context every greedy test diverged somewhere). With the model's default sampling the gain is smaller than with `temperature: 0`.


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
  on and off pass with the draft loaded. With the draft on and the model's default sampling, one request at a time
  (`research/2026-09-23-speculative-decoding/exp04-eval-with-draft/`): code 40/40, instruct 20/20, tools 10/10 —
  the same as without; 86% of drafted tokens accepted, 7 s per answer instead of 9.
- **Engine on mlx-vlm 0.6.16, verify pass as one plain forward** (2026-09-24, `research/2026-09-24-engine-upgrade-vlm616/`,
  `research/2026-09-23-speculative-decoding/exp11-plain-verify/`): the draft gives code 60.6 tok/s (1.53x) and copy-editing
  1.51x with output identical to plain decoding, at 16 and at 8 bits; a story diverges after a few dozen tokens (as with
  every engine version before). mlx-vlm's own bit-exact verifier was measured at 109 ms per 8-token block against 47 for a
  plain forward and is not used. Integration tests 5/5 with `kv_cache_bits: 8` and the draft; the on-disk cache written by
  the previous engine restores.
- **`draft: dflash2`** (2026-09-24, `research/2026-09-23-speculative-decoding/exp12-dflash2-in-engine/`, `exp14-dflash2-4bit/`):
  the drafter at 4 bits gives code 1.71x and copy-editing 2.05x, the bf16 original 1.51x and 1.83x — same tokens accepted
  per round, same output. Greedy code and copy-editing identical to plain decoding at 16 bits, a story diverges;
  with `kv_cache_bits: 8` code also diverges after some tokens. 81% of drafted tokens accepted on code with the model's
  default sampling. Integration tests 5/5 with the DFlash 2 draft and `kv_cache_bits: 8`. The drafter only sees the part of
  the prompt this request computed — a prefix that came back from the disk cache is not fed to it — which costs nothing
  measurable (research exp09: the last 256 tokens carry all of the acceptance rate).
- **Both together — `kv_cache_bits: 8` with `speculative_decoding: true`, the configuration recommended above** (2026-09-24,
  `research/2026-09-23-speculative-decoding/exp05-kv8-with-draft/`): the first request crashed on the engine as shipped
  (its verify step could not read a quantized cache; fixed in our engine fork). After the fix: greedy code and copy-editing
  output identical to plain 8-bit decoding at 1.49x, 86% of drafted tokens accepted with the model's default sampling,
  integration tests 5/5 with both on, including the prompt cache surviving a restart.


## Not tested

- Other Macs than the M3 Ultra 96 GB.
- `reasoning_effort` levels: wired through, not yet measured for speed or quality.
