# lmk (lm-kitten)

A local LLM server for kitten, built on the open-source
[mlx-engine](https://github.com/lmstudio-ai/mlx-engine). It replaces the closed
HTTP layer of LM Studio; the inference engine, the on-disk prefix cache and the
tool-call parsers are all open source and used as they are.

Design: `../docs/design/2026-09-19-lmk.md`. Why it exists:
`../research/2026-09-19-local-llm-server-wishlist/notes.md`.

## Run

    make venv        # python 3.11 env + mlx-engine at the commit in ENGINE_COMMIT
    make test        # unit tests — no GPU, no model
    make run         # start the server with ~/.kitten/lmk.yaml

`~/.kitten/lmk.yaml` (machine-level; `LMK_CONFIG=<path>` overrides the location):

    model:
      path: ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit
      context_length: 200000
      # id: the name clients send as "model"; defaults to the directory name, lowercased
    listen:
      port: 1235        # required — no default port
      # host: 127.0.0.1
    # cache: {dir: ~/.kitten/lmk/cache}
    # log:   {dir: ~/Library/Logs/kitten}

The model named there is loaded at startup and stays resident. There is no
just-in-time loading and no idle eviction.

The prefix cache is persistent: records live as files under `cache.dir`, in a
subdirectory keyed by the model's identity (weights + config + engine commit),
and the index is rebuilt from them at startup. A reboot no longer means a cold
first turn. Stop the server with SIGTERM/ctrl-c so queued records are flushed.

## Install as a resident service

    make install     # copies to ~/.kitten/lmk/app, builds its env, (re)starts LaunchAgent ai.kitten.lmk
    make uninstall   # stops it and removes the LaunchAgent; keeps the app dir, config and cache

The service runs from `~/.kitten/lmk/app`, never from this working tree — switching
branches in the repo must not take the server down. Rerun `make install` to deploy changes.
Logs: `~/Library/Logs/kitten/lmk.jsonl` (structured), `lmk.stderr.log` (the engine's own output).

## Endpoints

- `GET /lmk/v1/status` — the resident model, its context length, what is in flight
- `GET /v1/models` — OpenAI-shaped list containing exactly the resident model
- `POST /v1/chat/completions` — OpenAI-shaped chat, streaming or not, with tools. lmk's
  additions sit where the shape allows them, so stock OpenAI clients keep working:
  cache hits in `usage.prompt_tokens_details.cached_tokens`, reasoning in
  `delta.reasoning_content`, prefill progress as chunks with `choices: []` and an
  `lmk.prefill` object. Closing the connection cancels the call. Images: OpenAI
  `image_url` parts with inline `data:image/...;base64,...` URLs only — lmk never
  fetches a URL. Whether the resident model takes images is in the status reply
  (`model.input_modalities`).
- `POST /lmk/v1/warmup` — same body as a chat request; prefills the prefix into the
  cache and generates nothing. Send system + tools; a later chat that starts the
  same way restores it.

Optional request headers, logged per call and shown in `/lmk/v1/status`:
`X-Lmk-Purpose` (turn / compaction / groom / warmup …), `X-Lmk-Ref-Id` (the caller's
own reference for this call), `traceparent`.

## Upgrading the engine

`requirements.txt` is mlx-engine's own pinned file at the commit in
`ENGINE_COMMIT`. To upgrade: change the commit, copy that commit's
`requirements.txt` over this one, `make clean venv`, and rerun the integration
tests. It is a deliberate act, never a side effect.
