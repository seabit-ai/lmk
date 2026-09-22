# gemma-4-12b-4bit — tried, not listed

`lmstudio-community/gemma-4-12B-it-MLX-4bit` (commit f45bda56), Google's dense 12B Gemma 4 with vision,
4-bit. It is **not in lmk's tested list**: it passed the integration tests but not the agent tasks.
This page is here so the work is not repeated; it may earn its place with a newer quantization.

## What we saw (M3 Ultra 96 GB, 2026-09-22)

| | |
|---|---|
| integration tests | pass, thinking off and on (tool call round trip, cache hits, warm-up, images, cache across a restart) |
| memory when loaded | 6 GiB; context 262,144 on our Mac; by the formula about 85k on a 16 GB Mac |
| speed | 666 tok/s reading a new prompt, 52k tok/s from the cache, 72 tok/s writing |

The agent tasks are three small ones every listed model passes: read a file then edit it based on
the content, answer a question without calling a tool, read "the first 5 lines" (an integer
argument).

| run | read-then-edit | no-tool question | integer argument |
|---|---|---|---|
| temperature 0 | **looped**: after the tool result it emitted `<|channel>thought` 2,000 times and never called the edit | ok | **dropped `max_lines`** |
| model's own sampling, run 1 | ok | ok | dropped `max_lines` |
| run 2 | **looped** | ok | dropped `max_lines` |
| run 3 | ok | ok | ok |

## Why it is not listed

- **It loops after a tool result** — 1 run in 4. An agent that hangs on one step in four is not usable,
  and a loop burns the whole `max_tokens` before it fails.
- **It drops an argument** the request stated plainly, 3 runs in 4.
- The same tasks on the same day: Gemma 4 E4B (half the parameters) 4 runs of 4 correct; Gemma 4
  26B-A4B 1 of 1.

## If you want to try it anyway

```yaml
model:
  repo: lmstudio-community/gemma-4-12B-it-MLX-4bit
```

Then `lmk pull`, `lmk up`. lmk handles its template (the same Gemma 4 dialect as the listed models).
Slower to write than the 26B-A4B mixture of experts (72 against 120 tok/s): a dense 12B reads all
its weights for every token.

## Speed row (`lmk bench`)

| chip | memory | model | context | prefill | cached prefill | decode | lmk | engine | date |
|---|---|---|---|---|---|---|---|---|---|
| Apple M3 Ultra | 96 GB | gemma-4-12b-4bit | 262,144 | 666 tok/s (4,028 tokens) | 52k tok/s (3,840 cached; first token 0.51 s) | 71.8 tok/s | v0.5.0+ | 08f0c07 | 2026-09-22 |
