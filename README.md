# lmk

**Run Qwen3.8-27B on an Apple Silicon Mac, simplified.**

* Runs `Qwen3.8-27B-MLX-4bit` with its full `262k` context. Tested on my own `M3 Ultra 96G`.
* The prefill cache lives on `local disk` and `never expires` (evicted by LRU only when the disk budget is full). For my agents this is the feature that matters most.
* `OpenAI-compatible API`, pinned to one model (`~/.lmk/config.yaml`). No wondering whether a request loaded a second model by mistake.
* Sessions with a `shared prefix` share the cache — multiple agents on the same system prompt and tools, or a conversation `forked` or `rewound`.
* A step answers in `about a second` on a small cache miss — also after lmk has been restarted, because the cache is on disk, not in memory.
* Good visibility: `lmk status` shows every request in flight and where it is — starting, prefilling, decoding, waiting for its turn.
* Parallel requests, configurable, if you have the memory.
* `Speculative decoding` with the model's own draft head: Qwen3.8-27B writes code 1.5x faster (39 → 60 tokens/s) with the
  answer token for token the same, one line in the config. The draft head is in the original weights but not in any MLX
  conversion; `lmk up` fetches the one we split out and verified.
* `KV cache at 8 bits`, one line in the config: the 27B fits 122k tokens of context on a 32 GB Mac instead of 85k, and scored
  the same as 16-bit with 60k tokens of files in front of every task.
* Nine tested models today: Qwen3.8-27B at 4/5/6/8-bit, the Qwen3.5-122B mixture of experts in two sizes, and Gemma 4 in three
  sizes; any MLX model on HuggingFace can be configured, untested by us. `Wish list` items are welcome.
* Built on [mlx-engine](https://github.com/lmstudio-ai/mlx-engine). Huge thanks to the LM Studio and MLX teams.

Measured on an M3 Ultra (96 GB) with Qwen3.8-27B-MLX-4bit, one request at a time, from lmk's own
request log (`LmkChatDone`) and [`research/2026-09-20-memory-guard`](research/2026-09-20-memory-guard/notes.md) MG-009:

| | tokens/s | what a step feels like |
|---|---|---|
| prefill, cache hit (restored from disk) | **~65,000** (about 75 ms + 10 ms per 1k tokens) | a 27k conversation resumes in 0.4 s, a 70k one in 1 s |
| prefill, cache miss (computed) | **~320** on an empty context; ~200 when appended to 50k+ cached tokens | an 11k system prompt costs 37 s — once |
| decode | **~33** (27 at 60k+ context) | |

The cache-hit numbers are with the cache files warm in the OS page cache; a true cold read after a
reboot is not measured yet.

`lmk bench` measures the same things on your Mac — prefill, cache hit, decode on prose and on code — says which
switches were on, and prints a row for [`docs/benchmarks.md`](docs/benchmarks.md); other machines and models
are what that table is missing. It starts with a canary: a fixed prompt whose answer it checks, because an engine
that produces wrong text on some Mac would otherwise bench beautifully. If anything looks off, `lmk report` prints
one Markdown block — this Mac, build, configuration, status, the canary answer, the last events and traceback —
to paste into an [issue](https://github.com/seabit-ai/lmk/issues/new). Nothing is sent anywhere by itself.

### Why not the server you already have?

Same Mac, same model, same requests (September 2026; method and raw numbers in
[`research/2026-09-20-local-server-survey`](research/2026-09-20-local-server-survey/notes.md), in Chinese):

| | next step of a conversation | after the server restarts |
|---|---|---|
| **lmk** | **0.8 – 1.5 s** | 2.4 s |
| oMLX 0.7.0.dev4 | 9.8 – 10.0 s | 12.9 s |
| LM Studio | 0.8 – 1.5 s | starts over: its cache does not survive a model reload |

* LM Studio's cache is gone after a model reload, a process restart or a reboot.
* oMLX caches in 4,096-token blocks, so a step re-reads up to 4,095 tokens. At ~320 tokens/s cold prefill on my
  M3 Ultra that is about 7 s per step on average. lmk saves a resume point at the end of every request
  (256-token granularity), so each step continues from where the last one stopped. With my agent the
  difference is 1–2 s against 10–20 s per step, and an agent loop makes many steps per task.

## Install

You need an Apple Silicon Mac and `git`. No Python required — the installer brings its own (venv).
The default model takes about 16 GB of memory for its weights, more as conversations grow, and
16 GB of disk. We have run lmk on one machine so far — an M3 Ultra with 96 GB — so how it behaves
on a smaller/larger Mac is something you may want to leave feedback to let us know.

```sh
brew install seabit-ai/tap/lmk
lmk up
lmk bench
```

Without Homebrew: `git clone https://github.com/seabit-ai/lmk && cd lmk && ./install.sh` (or
`curl -fsSL https://raw.githubusercontent.com/seabit-ai/lmk/main/install.sh | sh`). Either way lmk
lives under `~/.lmk` — its own Python, the engine, the config, the cache; Homebrew holds only the
command — and the first `lmk` command sets that up (about 1 GB, a minute). `lmk up` downloads the
model if it is not on this Mac yet (16 GB, once, into the shared HuggingFace cache; one progress line
with the percentage, the rate and the time left; interrupt any time, it resumes), then starts lmk now
and at every login. It returns when the model is loaded and has answered a test request, and prints
what to paste into your agent:

```
✓ lmk is up    http://127.0.0.1:1235/v1   (OpenAI-compatible)
  model      qwen3.8-27b-4bit · text, image in · 262,144 tokens
  settings   thinking off · KV cache 8-bit · speculative decoding on
  draft      no tokens drafted yet
  about it   https://github.com/seabit-ai/lmk/blob/main/docs/models/qwen3.8-27b-4bit.md
  sampling   temp 1.0 · top_p 0.95 · top_k 20   (the model's generation_config; a request may override)
  memory     pressure: normal · 76% of 96.0 GB free · lmk holds 16.5 GB
  cache      0 B of 200.0 GB in ~/.lmk/cache
  requests   answering 0 of 2 · waiting 0 of 16 · tokens in memory 0 of 1,844,474
  since start  0 answered · 0 refused · 0 failed · 0 cancelled · up 1s · build v0.7.0

  idle — no requests

  Point your agent at it — any OpenAI-compatible client:
    base URL   http://127.0.0.1:1235/v1
    model      qwen3.8-27b-4bit
    API key    anything (lmk does not check it)
```

Below that it prints the same thing as a ready-to-paste config block for [OpenClaw](https://github.com/openclaw/openclaw)
and for kitten, with this machine's address, model id and context size filled in.

See it answer:

```sh
curl http://127.0.0.1:1235/v1/chat/completions \
  -d '{"model":"qwen3.8-27b-4bit","messages":[{"role":"user","content":"Reply with one word: ready"}]}'
```

Everything lmk installs lives in `~/.lmk`. The model goes to the shared HuggingFace cache
(`~/.cache/huggingface/hub`), where your other tools can use it too.

## The whole command line

| | |
|---|---|
| `lmk pull` | Download the configured model now, to start later. `lmk up` downloads it too when it is missing; nothing else ever downloads anything. |
| `lmk up` | Start lmk, now and at every login. Run it again after changing the config or upgrading. |
| `lmk status` | Is it up, what is it doing right now, how full is the cache. |
| `lmk logs` | Recent events. `-f` to follow, `--raw` for the model runtime's own output. |
| `lmk down` | Stop it, and don't start it at login. Model, cache and config are kept. |
| `lmk bench` | A canary answer check, then prefill, cache-hit and decode speed (prose and code) on this Mac, with the switches in effect, as a row for [`docs/benchmarks.md`](docs/benchmarks.md). |
| `lmk report` | One Markdown block for a GitHub issue: this Mac, build, configuration, status, the canary answer, the last events and traceback. |

When a request seems stuck, `lmk status` shows what it is doing:

```
  busy       1 request
               reading prompt 10,240 / 26,938 (38%) · turn · my-session/step-4 · 37s
```

## Models

Tested end to end with a real agent — tool calls, thinking, images and the on-disk prompt cache — on an
M3 Ultra with 96 GB. **Each model has its own page** with what fits, how fast, the recommended
configuration and what to know before using it. Put the name under `model.name` in `~/.lmk/config.yaml`;
`lmk up` downloads it.

The groups say which Mac a model runs on, and "ctx size" the context that fits in that Mac's memory —
how long a conversation can get (1k tokens is about 750 words of English; an agent's system prompt
alone is often 10k). Both come from the runtime's own memory formula and what each model took on our 96 GB Mac, not
from running on those Macs — if you have one, `lmk bench` and an issue with its row would tell
everyone. Speeds are tokens per second on the M3 Ultra ([`docs/benchmarks.md`](docs/benchmarks.md)):
reading a prompt that is in the cache, reading a new one, and writing the answer. What we have not
measured is how smart each model is; the default is the one we have used most.

<!-- models-table -->
### Needs at least 16 GB

| model | good for | ctx size on 16 GB / 24 GB | images | tok/s: cache hit / miss / decode |
|---|---|---|---|---|
| [`gemma-4-e4b-4bit`](docs/models/gemma-4-e4b-4bit.md) | the small one; 7 GB | 75k / 131k tokens | yes | 183k / 2,199 / 94 |

### Needs at least 24 GB

| model | good for | ctx size on 24 GB / 32 GB / 48 GB | images | tok/s: cache hit / miss / decode |
|---|---|---|---|---|
| [`gemma-4-26b-a4b-4bit`](docs/models/gemma-4-26b-a4b-4bit.md) | fastest here by far; MoE, 4B active | 46k / 210k / 262k tokens | yes | 83k / 1,833 / 120 |

### Needs at least 32 GB

| model | good for | ctx size on 32 GB / 48 GB / 64 GB / 96 GB | images | tok/s: cache hit / miss / decode |
|---|---|---|---|---|
| [`qwen3.8-27b-4bit`](docs/models/qwen3.8-27b-4bit.md) (default) | the default; best-tested with `reasoning_effort: low` | 122k (85k at 16-bit) / 262k (223k at 16-bit) / 262k / 262k tokens | yes | 53k / 323 / 40 |
| [`qwen3.8-27b-5bit`](docs/models/qwen3.8-27b-5bit.md) | the 27B between 4- and 8-bit: 19 GB, 20% slower decode than 4-bit | 51k / 190k / 262k / 262k tokens | yes | 57k / 315 / 32 |
| [`gemma-4-31b-4bit`](docs/models/gemma-4-31b-4bit.md) | Gemma at the 27B's size; slower, shorter ctx | 39k / 150k / 261k / 262k tokens | yes | 26k / 252 / 33 |

### Needs at least 48 GB

| model | good for | ctx size on 48 GB / 64 GB / 96 GB | images | tok/s: cache hit / miss / decode |
|---|---|---|---|---|
| [`qwen3.8-27b-8bit`](docs/models/qwen3.8-27b-8bit.md) | the 27B with less quantization loss, 40% slower decode | 89k / 228k / 262k tokens | yes | 44k / 319 / 23 |
| [`qwen3.8-27b-6bit`](docs/models/qwen3.8-27b-6bit.md) | the 27B at 6-bit: 23 GB, 30% slower decode than 4-bit | 156k / 262k / 262k tokens | yes | 53k / 315 / 28 |

### Needs at least 64 GB

| model | good for | ctx size on 64 GB / 96 GB | images | tok/s: cache hit / miss / decode |
|---|---|---|---|---|
| [`qwen3.5-122b-a10b-48gb`](docs/models/qwen3.5-122b-a10b-48gb.md) | the 122B for 64 GB Macs (2–3 bit experts, community quant) | 83k / 262k tokens | yes | 89k / 746 / 54 |

### Needs at least 96 GB

| model | good for | ctx size on 96 GB / 128 GB | images | tok/s: cache hit / miss / decode |
|---|---|---|---|---|
| [`qwen3.5-122b-a10b-4bit`](docs/models/qwen3.5-122b-a10b-4bit.md) | the biggest here; MoE, faster than the 27B but not smarter on our tests | 165k / 262k tokens | yes | 89k / 753 / 60 |

Where a model's recommended `kv_cache_bits: 8` changes the number, the figure at the model's own 16-bit precision is in parentheses; its page says what the setting costs.
<!-- /models-table -->

Any other MLX model on HuggingFace loads through `model.repo` (see Configuration), untested by us.
The runtime keeps its prompt cache on disk only for models whose config has a `vision_config`;
a text-only model runs, but every restart starts cold. Speed on other Macs: [`docs/benchmarks.md`](docs/benchmarks.md).

## What lmk is not

lmk runs **one model per machine**, and that is the point. There is no model library to browse, no
second model loaded on the side, no per-model settings panel, no menu-bar app, no web console, and
it never updates itself. If you want to try many models, use a tool made for that. If you have
picked a model and want your agent to be fast on it every day, that is what lmk is for.

## What does not work yet

- **`seed` is ignored** — the engine drops it on the batched code path lmk runs on. For a repeatable answer
  send `temperature: 0`. `response_format` / JSON schema output is not wired up yet.
- Five tested models (see Models). Others load through `model.repo`, untested by us.
- The memory rules below are tested on one machine (96 GB), where most of them never trigger; on a smaller Mac
  they are covered by unit tests only.
- No PDF input.

## Configuration

There is nothing you have to configure. `~/.lmk/config.yaml` is written at install with every value in use, so what you see is what runs.
`~/.lmk/config.yaml.example` next to it is a config that runs too — the defaults, plus the three settings the default
model's page recommends — with every tested model as a line to uncomment and every other choice as a commented
example; lmk rewrites it at each `lmk up`, so it always matches the installed version. The settings, with their defaults:

```yaml
model:
  name: qwen3.8-27b-4bit     # a tested model (see Models). Clients send that name as "model"
  # repo: mlx-community/Qwen3-30B-A3B-4bit   # instead of name: any MLX model on HuggingFace (its address after
                             #                huggingface.co/); untested by us. Clients send its last part, lower case
  # path: ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit   # instead of name: an MLX model folder
                             #                already on this Mac — here, one LM Studio downloaded
  # thinking: false           # the model answers without thinking (default: the template's own, on for Qwen)
  # reasoning_effort: low     # for templates that know it (Qwen3.8: low / medium / xhigh); a server-wide constant
  # kv_cache_bits: 8          # 8 halves what each token of context costs in memory (about twice the context on the
                             # same Mac); the model page says whether we tested it. Default 16, the model's own precision
  # speculative_decoding: true  # the model's own draft head guesses tokens, the model checks them: same answers on
                             # code, faster while one request is being answered. Needs the draft `lmk pull` fetches;
                             # the model page says whether there is one. Default false
  # context_length: 65536     # default: the model's maximum; lmk lowers it if memory is short, and `lmk status`
                             # shows the value in use
listen: {host: 127.0.0.1, port: 1235}
cache:  {dir: ~/.lmk/cache, max_size: 200G}     # when full, what was used longest ago goes first
requests:
  max_parallel: 2            # answered at the same time
  max_queue: 16              # waiting for their turn; one more is refused at once
  max_wait_seconds: 600      # a request that could not start by then is refused, and told why
log:    {dir: ~/.lmk/logs}
```

`max_size` is the limit. One guard on top of it: when the disk has less than 10 GB free, lmk stops
adding to the cache and gives space back, oldest first.

To switch models: change `model:`, then `lmk up` (it downloads the new model first).

## Several requests at once, and memory

What answering several requests at once gains depends on how long the conversations are.
Measured on an M3 Ultra with the default model:

| requests at once | prompts of a few dozen tokens | 27k-token conversations |
|---|---|---|
| 1 | 40 tokens/s | 33 tokens/s |
| 2 | 34 each — 1.7x in total | 16 each — 1.0x in total |
| 4 | 22 each — 2.2x in total | not measured |

With long conversations — an agent's usual case — a single request already keeps the GPU busy, so
`max_parallel: 2` changes the shape of the wait rather than the total: a second request starts
answering at once, at half speed, instead of waiting for the first to finish. Set it to 1 to have
each request at full speed, one after the other. Reading a long new prompt is a different matter:
it stalls every request that is writing.
So lmk keeps one first-come-first-served queue, and the request at its head starts when:

- fewer than `max_parallel` requests are being answered;
- nobody is being answered, **or** it has little new prompt to read — the next step of a cached
  conversation always goes straight in, however long the conversation is;
- its tokens fit in memory next to the requests already running;
- macOS does not report memory pressure as critical.

`lmk status` shows who waits and for what, and the memory numbers lmk goes by:

```
  memory     pressure: normal · 64% of 96.0 GB free · lmk holds 15.2 GB
  busy       1 request
               writing the answer · turn · my-session/step-4 · 4s
  waiting    1 request
               groom · my-project/groom · 2s · another request is being answered, and this one has 14,061 tokens of new prompt to read first
```

Nothing is sent to a waiting client, so one that cannot start within `max_wait_seconds`, or finds
the queue full, gets a plain `503` whose message says what it was waiting for.

A model whose weights do not fit this Mac is refused by `lmk up`, with the numbers. There is no
switch to load it anyway.

## For agent authors

lmk follows the OpenAI shape and puts its additions where that shape has room, so stock clients
keep working and yours can do better:

- **Cache hits** are reported per request in `usage.prompt_tokens_details.cached_tokens`.
  The usage chunk also carries `lmk.restore_ms` (how long the cached part took to come back from disk) and `lmk.first_token_ms`.
- **Thinking** arrives separately, in `delta.reasoning_content`.
- **Tool calls** come back as structured `tool_calls` with JSON arguments, whatever format the model writes natively.
- **Prompt-reading progress**: while a long prompt is being read, the stream carries chunks with
  `choices: []` and `lmk.prefill: {processed, total, cached}`. Draw a progress bar — or ignore them;
  they also keep the connection from timing out. Use `stream: true` for long prompts.
- **Images**: OpenAI `image_url` parts with inline `data:image/...;base64,` URLs. lmk never fetches a URL.
- **Say who is calling**: optional headers `X-Lmk-Purpose` and `X-Lmk-Ref-Id` show up in
  `lmk status` and in the logs, next to the request they belong to.
- **Warm a prompt ahead of time**: `POST /lmk/v1/warmup` takes a chat request body, reads the
  prompt into the cache and generates nothing.
- **Sampling**: `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` and `stop` are honoured;
  a value out of range is a 400 naming the field. A request that sets none of them runs with the
  model's own `generation_config.json` (`lmk status` shows those values). `stop` strings match the
  **answer only** — a stop string that shows up inside the model's thinking does not end the request.
  `seed` is accepted and ignored (logged as `LmkParamIgnored`); see "What does not work yet".
- Closing the connection cancels the request (at the next progress step — within a few seconds). `GET /lmk/v1/status` is what `lmk status` prints.


## Why it is fast

Reading a prompt is the slow part of running a large model locally, and an agent re-sends almost
the same prompt at every step. lmk stores what the model computed for each prompt on disk, in
small blocks addressed by their content, so anything that starts the same way as something seen
before — the next step of a conversation, a new conversation with the same system prompt and
tools — skips straight to the new part. The store survives restarts and is shared across
conversations.

Writing the answer is the other half. With `speculative_decoding: true` a small draft — Qwen3.8's own
multi-token-prediction head, which the original weights ship and the MLX conversions drop — guesses the
next few tokens and the model checks them in one pass. Guesses it agrees with are free; on code and
copy-editing the answer is token for token the one the model would have written alone, on prose it can
differ after a few dozen tokens (greedy is exact up to floating-point ties). On the M3 Ultra the 27B goes
from 39 to 60 tokens/s on code and copy-editing, with 86–89% of its guesses accepted across our agent
tests, and to 45 on prose.
It runs while one request is being answered; several at once are decoded plainly.
`lmk up` fetches the draft for the models that have one; the model page says so.

The model runtime is [mlx-engine](https://github.com/lmstudio-ai/mlx-engine), the open-source
engine behind LM Studio, run from lmk's own fork ([seabit-ai/mlx-engine](https://github.com/seabit-ai/mlx-engine),
branch `lmk`): the upstream code at a pinned commit plus a short list of patches, one commit each — so far, KV
cache quantization and speculative decoding on the batched path, the one with the on-disk prompt cache.
How lmk compares with other servers on the same machine,
with the raw numbers: [`research/2026-09-20-local-server-survey`](research/2026-09-20-local-server-survey/notes.md)
(notes are in Chinese).

## Upgrading and removing

Upgrade: `brew upgrade lmk && lmk up` (the next `lmk` command installs the new build into `~/.lmk`),
or from a clone `git pull && ./install.sh && lmk up`. Your cache is kept across upgrades.

`install.sh` run from a clone installs that clone. Run without one
(`curl -fsSL https://raw.githubusercontent.com/seabit-ai/lmk/main/install.sh | sh`) it installs
the newest [release](https://github.com/seabit-ai/lmk/releases); `LMK_REF=main` (or a tag, or a
commit) picks something else. `lmk status` shows which build is running.

Remove: `lmk down`, then delete `~/.lmk` (and `brew uninstall lmk`). The model stays in the HuggingFace cache until you
delete it there (`hf cache rm`, or remove its folder under `~/.cache/huggingface/hub`).

## Working on lmk

```sh
make venv      # python 3.11 env + mlx-engine at the commit in ENGINE_COMMIT
make test      # unit tests — no GPU, no model
make itest     # integration tests — load the configured model
make install   # install this working tree into ~/.lmk and restart the service
```

CI (`.github/workflows/test.yml`) runs the unit tests, lint and `install.sh` on an Apple Silicon
runner; the integration tests need the model and a GPU and stay on a developer's Mac.

How we work here, the map of the code and the traps already hit are in [`CLAUDE.md`](CLAUDE.md); what is open,
unverified or decided-but-not-built is in [`docs/backlog.md`](docs/backlog.md). Design notes are in
[`docs/design`](docs/design) (all three in Chinese). Upgrading mlx-engine is a deliberate
act: run `make cache-fixture` first, change `ENGINE_COMMIT` and copy that commit's
`requirements.txt` over ours, `make clean venv`, then `make cache-compat` and `make itest`. If
`cache-compat` fails, the new engine cannot read the old cache: bump `CACHE_FORMAT_VERSION`.

## License

MIT — see [LICENSE](LICENSE). The model runtime lmk installs,
[mlx-engine](https://github.com/lmstudio-ai/mlx-engine), is MIT as well.
