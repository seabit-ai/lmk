"""The two config files lmk writes (design OOBE §C, the same scheme as kitten's):

  config.yaml          seeded once as a REAL config — every value written out, nothing
                       commented away (owner, 2026-09-22) — then never touched: it is the user's.
  config.yaml.example  ours: rewritten whenever its content differs, so it is always the current
                       full reference. A REAL config too (owner, 2026-09-24): the defaults plus the
                       default model's recommended settings, every other choice as a commented line.
"""
import os
import textwrap
from pathlib import Path

from lmk.config import (DEFAULT_CACHE_MAX_SIZE, lmk_home, DEFAULT_HOST, DEFAULT_MAX_PARALLEL, DEFAULT_MAX_QUEUE,
                        DEFAULT_MAX_WAIT_SECONDS, DEFAULT_PORT)
from lmk.models import DEFAULT_MODEL_NAME, TESTED_MODELS, smallest_mac_gb

def _seed_text() -> str:
    home = _home_for_humans()
    return textwrap.dedent(f"""\
    # lmk configuration — these are the values in use. After a change: `lmk up` (it restarts the service).
    # The full reference, with every option and the tested models, is config.yaml.example next to this file.

    model:
      name: {DEFAULT_MODEL_NAME}       # a tested model; the list is in config.yaml.example
      # For a model that is not in the tested list, replace the name: line with ONE of these:
      # repo: mlx-community/Qwen3-30B-A3B-4bit   # its HuggingFace address after huggingface.co/ (any MLX model;
      #                                          # `lmk pull` downloads it; whether it works, we have not checked)
      # path: /Users/me/models/Some-Model-MLX    # a model folder already on this Mac (nothing to download)
      # context_length: 131072               # default: the model's own maximum, lowered if memory is short
      # thinking: false                      # answer without thinking (default: the template's own — on for Qwen)
      # reasoning_effort: low                # low / medium / xhigh where the model's template knows it; server-wide,
      #                                      # never per request (it sits at the start of every prompt)
      # kv_cache_bits: 8                     # 8 halves what each token of context costs in memory: about twice the
      #                                      # context on the same Mac, answers may differ slightly on very long
      #                                      # prompts. Default 16 (the model's own precision); 4 is accepted too
      # speculative_decoding: true           # a small draft guesses the next tokens, the model checks them: the same
      #                                      # answers on code, faster while one request is being answered (several at
      #                                      # once are answered plainly); prose may differ slightly. Needs the draft
      #                                      # `lmk pull` fetches — the model page says whether there is one
      # draft_tokens: 3                      # advanced: tokens the draft guesses per round (default: the draft's own)

    listen:
      host: {DEFAULT_HOST}
      port: {DEFAULT_PORT}

    cache:                        # prompts already processed, kept on disk: a conversation continues
      dir: {home}/cache           # in about a second — also after a reboot
      max_size: {DEFAULT_CACHE_MAX_SIZE}              # when full, what was used longest ago is dropped first

    requests:
      max_parallel: {DEFAULT_MAX_PARALLEL}             # answered at the same time (see config.yaml.example for what it gains)
      max_queue: {DEFAULT_MAX_QUEUE}               # waiting for their turn; one more is refused at once
      max_wait_seconds: {DEFAULT_MAX_WAIT_SECONDS}       # a request that could not start by then is refused, and told why

    log:
      dir: {home}/logs
    """)


def _model_lines() -> str:
    """One commented `name:` line per tested model, grouped by the smallest Mac it should run on:
    choosing a model = uncommenting a line. Expected from memory measured on a 96 GB Mac."""
    tiers: dict[int, list[str]] = {}
    for name, m in TESTED_MODELS.items():
        tiers.setdefault(smallest_mac_gb(m), []).append(name)
    lines = []
    for gb in sorted(tiers):
        lines.append(f"  #   -- a Mac with at least {gb} GB --")
        for name in tiers[gb]:
            m = TESTED_MODELS[name]
            lines.append(f"  # name: {name:<24} # {m.size_gb:.1f} GB download, {m.loaded_gib:.0f} GiB loaded: {m.good_for}")
    return "\n".join(lines)


def _draft_family() -> str:
    """The tested models that have a draft head, as one phrase: `the qwen3.8-27b ones`."""
    names = [n for n, m in TESTED_MODELS.items() if m.draft_repo]
    prefix = os.path.commonprefix(names).rstrip("-")
    return f"the {prefix} ones" if len(names) > 1 and prefix else ", ".join(names)


def example_text() -> str:
    home = _home_for_humans()
    return f"""\
# config.yaml.example — every setting, as a config that runs. lmk rewrites this file at each `lmk up` so it always
# matches the installed version: do not edit it; copy it, or the lines you want, into config.yaml next to it,
# then `lmk up`. The values are the defaults, except the three the {DEFAULT_MODEL_NAME} page recommends.
# What each setting does and what we measured: README.md "Configuration" and docs/models/<name>.md.

model:
  name: {DEFAULT_MODEL_NAME:<24} # a tested model, one of these:
{_model_lines()}
  # repo: mlx-community/Qwen3-30B-A3B-4bit   # instead of name: any MLX model on HuggingFace (its address after
  #                                          # huggingface.co/); `lmk pull` downloads it; untested by us
  # path: /Users/me/models/Some-Model-MLX    # instead of name: a model folder already on this Mac
  # The next three are the {DEFAULT_MODEL_NAME} page's recommendation; for another model its page says which to keep.
  reasoning_effort: low          # Qwen3.8: low / medium / xhigh (other models: their page); server-wide
  # reasoning_effort: xhigh      # the template's own default; scored worse than low on every test
  kv_cache_bits: 8               # 16 = the model's own precision; 8 = about twice the context on the same Mac
  # kv_cache_bits: 16            # for a model whose page does not recommend 8
  speculative_decoding: true     # the model's own draft head guesses tokens, the model checks them (`lmk pull` fetches it)
  # speculative_decoding: false  # for a model without a draft head: today all but {_draft_family()}
  # thinking: false              # answer without thinking; default: the template's own (on for Qwen, off for Gemma)
  # context_length: 65536        # default: the model's maximum, lowered if memory is short; `lmk status` shows the value in use
  # draft_tokens: 3              # advanced: tokens the draft guesses per round (default: the draft's own)

listen:
  host: {DEFAULT_HOST}
  port: {DEFAULT_PORT}
  # port: 8080

cache:
  dir: {home}/cache              # processed prompts kept on disk: a conversation continues in about a second, also after a reboot
  max_size: {DEFAULT_CACHE_MAX_SIZE}                 # when full, what was used longest ago is dropped first
  # max_size: 50G

requests:
  max_parallel: {DEFAULT_MAX_PARALLEL}                # answered at the same time (README: what it gains, measured)
  # max_parallel: 1              # one at a time, each at full speed
  max_queue: {DEFAULT_MAX_QUEUE}                  # waiting for their turn; one more is refused at once
  max_wait_seconds: {DEFAULT_MAX_WAIT_SECONDS}          # a request that could not start by then is refused, and told why

log:
  dir: {home}/logs
"""


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _home_for_humans() -> str:
    """`~/.lmk` when it is the default home, else the real path (a test or trial install)."""
    home = lmk_home()
    return "~/.lmk" if home == Path.home() / ".lmk" else str(home)


def seed_config(path: Path) -> bool:
    """Returns True when the file was created."""
    if path.exists():
        return False
    _write_atomically(path, _seed_text())
    return True


def refresh_example(config_path: Path) -> bool:
    """Compares content, not mtime: a no-op start leaves the file untouched."""
    example = config_path.with_name(config_path.name + ".example")
    want = example_text()
    if example.exists() and example.read_text() == want:
        return False
    _write_atomically(example, want)
    return True
