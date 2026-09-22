# lmk

**One local model, always on, for your agent — on an Apple Silicon Mac.**

Agents don't send one prompt; they send the same growing conversation dozens of times, once per
tool call. lmk keeps every prompt it has processed on disk, so each step of a conversation starts
answering in **about a second** — also after lmk, or the Mac, has been restarted.

Measured on an M3 Ultra with Qwen3.8-27B and a real agent's requests (56 tools, 11k-token system
prompt, a 27k-token conversation):

| | time to first token |
|---|---|
| next step of a running conversation | **0.8 – 1.5 s** |
| first request after lmk restarts | 2.4 s |
| a prompt lmk has never seen | ~3 s per 1,000 tokens (37 s for that 11k prompt) — once |

It speaks the OpenAI chat-completions API, so any agent that can talk to OpenAI can talk to lmk.

### Why not the server you already have?

Same Mac, same model, same requests (September 2026; method and raw numbers in
[`research/2026-09-20-local-server-survey`](research/2026-09-20-local-server-survey/notes.md), in Chinese):

| | next step of a conversation | after the server restarts |
|---|---|---|
| **lmk** | **0.8 – 1.5 s** | 2.4 s |
| oMLX 0.7.0.dev4 | 9.8 – 10.0 s | 12.9 s |
| LM Studio | fast while the model stays loaded | starts over: its cache does not survive a model reload |

oMLX also keeps its cache on disk across restarts, and it does far more than lmk (many models,
an Anthropic API, a menu-bar app). The gap above is specific to models like Qwen3.5/3.8, which mix
attention with a recurrent state: for those oMLX caches in 4,096-token blocks, so every step
re-reads up to 4,095 tokens. lmk saves a resume point at the end of every request, so a conversation
continues from where it stopped. On a plain-attention model we expect the gap to disappear; we
have not measured that.

## Install

You need an Apple Silicon Mac and `git`. No Python required — the installer brings its own.
The default model takes about 16 GB of memory for its weights, more as conversations grow, and
16 GB of disk. We have run lmk on one machine so far — an M3 Ultra with 96 GB — so how it behaves
on a smaller Mac is not something we can promise yet.

```sh
git clone https://github.com/seabit-ai/lmk && cd lmk && ./install.sh
lmk pull
lmk up
```

`./install.sh` takes about 20 seconds and puts everything under `~/.lmk`. `lmk pull` downloads
the model (16 GB, once; it resumes if interrupted). `lmk up` starts lmk now and at every login; it
returns when the model is loaded and has answered a test request, and prints what to paste into
your agent:

```
✓ lmk is up    http://127.0.0.1:1235/v1   (OpenAI-compatible)
  model      qwen3.8-27b-4bit   (text, image in)
  context    262,144 tokens
  cache      0 B of 200.0 GB   ~/.lmk/cache
  running    1s   (build 908b57c)
  busy       no — idle

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
| `lmk pull` | Download the configured model. Nothing else ever downloads anything. |
| `lmk up` | Start lmk, now and at every login. Run it again after changing the config or upgrading. |
| `lmk status` | Is it up, what is it doing right now, how full is the cache. |
| `lmk logs` | Recent events. `-f` to follow, `--raw` for the model runtime's own output. |
| `lmk down` | Stop it, and don't start it at login. Model, cache and config are kept. |

When a request seems stuck, `lmk status` shows what it is doing:

```
  busy       1 request
               reading prompt 10,240 / 26,938 (38%) · turn · my-session/step-4 · 37s
```

## What lmk is not

lmk runs **one model per machine**, and that is the point. There is no model library to browse, no
second model loaded on the side, no per-model settings panel, no menu-bar app, no web console, and
it never updates itself. If you want to try many models, use a tool made for that. If you have
picked a model and want your agent to be fast on it every day, that is what lmk is for.

## What does not work yet

- **Sampling parameters are ignored** — `temperature`, `top_p`, `seed`, `stop`. The model's own defaults apply.
- One tested model (Qwen3.8-27B, 4-bit). Others load through `model.repo`, untested by us.
- The memory rules below are tested on one machine (96 GB), where most of them never trigger; on a smaller Mac
  they are covered by unit tests only.
- No PDF input.

## Configuration

There is nothing you have to configure. `~/.lmk/config.yaml` starts out as comments only;
`~/.lmk/config.yaml.example` next to it is the full, always-current reference — including the
list of models we have tested. The settings, with their defaults:

```yaml
model:
  name: qwen3.8-27b-4bit          # a tested model; or  repo: <any MLX model on HuggingFace>
                             #                 or  path: <a directory on this disk>
  # id: what clients send as "model"            (default: the name)
  # context_length:                             (default: the model's maximum; lmk lowers it if
                             #                   memory is short, and `lmk status` shows the value in use)
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

To switch models: change `model:`, then `lmk pull` and `lmk up`.

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
- Closing the connection cancels the request (at the next progress step — within a few seconds). `GET /lmk/v1/status` is what `lmk status` prints.


## Why it is fast

Reading a prompt is the slow part of running a large model locally, and an agent re-sends almost
the same prompt at every step. lmk stores what the model computed for each prompt on disk, in
small blocks addressed by their content, so anything that starts the same way as something seen
before — the next step of a conversation, a new conversation with the same system prompt and
tools — skips straight to the new part. The store survives restarts and is shared across
conversations.

The model runtime is [mlx-engine](https://github.com/lmstudio-ai/mlx-engine), the open-source
engine behind LM Studio, used as it is. How lmk compares with other servers on the same machine,
with the raw numbers: [`research/2026-09-20-local-server-survey`](research/2026-09-20-local-server-survey/notes.md)
(notes are in Chinese).

## Upgrading and removing

Upgrade: `git pull && ./install.sh && lmk up`. Your cache is kept across upgrades.

Remove: `lmk down`, then delete `~/.lmk`. The model stays in the HuggingFace cache until you
delete it there (`hf cache rm`, or remove its folder under `~/.cache/huggingface/hub`).

## Working on lmk

```sh
make venv      # python 3.11 env + mlx-engine at the commit in ENGINE_COMMIT
make test      # unit tests — no GPU, no model
make itest     # integration tests — load the configured model
make install   # install this working tree into ~/.lmk and restart the service
```

How we work here, the map of the code and the traps already hit are in [`CLAUDE.md`](CLAUDE.md); what is open,
unverified or decided-but-not-built is in [`docs/backlog.md`](docs/backlog.md). Design notes are in
[`docs/design`](docs/design) (all three in Chinese). Upgrading mlx-engine is a deliberate
act: run `make cache-fixture` first, change `ENGINE_COMMIT` and copy that commit's
`requirements.txt` over ours, `make clean venv`, then `make cache-compat` and `make itest`. If
`cache-compat` fails, the new engine cannot read the old cache: bump `CACHE_FORMAT_VERSION`.

## License

MIT — see [LICENSE](LICENSE). The model runtime lmk installs,
[mlx-engine](https://github.com/lmstudio-ai/mlx-engine), is MIT as well.
