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

## Endpoints

- `GET /lmk/v1/status` — the resident model, its context length, what is in flight
- `GET /v1/models` — OpenAI-shaped list containing exactly the resident model

## Upgrading the engine

`requirements.txt` is mlx-engine's own pinned file at the commit in
`ENGINE_COMMIT`. To upgrade: change the commit, copy that commit's
`requirements.txt` over this one, `make clean venv`, and rerun the integration
tests. It is a deliberate act, never a side effect.
