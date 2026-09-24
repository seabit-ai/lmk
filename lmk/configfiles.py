"""The two config files lmk writes (design OOBE §C, the same scheme as kitten's):

  config.yaml          seeded once as a REAL config — every value written out, nothing
                       commented away (owner, 2026-09-22) — then never touched: it is the user's.
  config.yaml.example  ours: rewritten whenever its content differs, so it is always the current
                       full reference. A REAL config too (owner, 2026-09-24): the defaults plus the
                       default model's recommended settings, every other choice as a commented line.
"""
import os
from pathlib import Path

from lmk.config import (DEFAULT_CACHE_MAX_SIZE, lmk_home, DEFAULT_HOST, DEFAULT_MAX_PARALLEL, DEFAULT_MAX_QUEUE,
                        DEFAULT_MAX_WAIT_SECONDS, DEFAULT_PORT)
from lmk.models import DEFAULT_MODEL_NAME, TESTED_MODELS, smallest_mac_gb

# One line per setting, shared by the seed and the example so the two never say different things.
# What each setting does and what we measured lives in README "Configuration" and the model pages, not here.
_WHAT = {
    "repo": "instead of name: any MLX model on HuggingFace (address after huggingface.co/); untested by us",
    "path": "instead of name: an MLX model folder already on this Mac — here, one LM Studio downloaded",
    "reasoning_effort": "Qwen3.8: low / medium / xhigh (other models: their page); server-wide",
    "kv_cache_bits": "16 = the model's own precision; 8 = about twice the context on the same Mac",
    "speculative_decoding": "the model's own draft head guesses tokens, the model checks them (`lmk pull` fetches it)",
    "thinking": "answer without thinking; default: the template's own (on for Qwen, off for Gemma)",
    "context_length": "default: the model's maximum, lowered if memory is short; `lmk status` shows the value in use",
    "draft_tokens": "advanced: tokens the draft guesses per round (default: the draft's own)",
    "cache_dir": "processed prompts kept on disk: a conversation continues in about a second, also after a reboot",
    "max_size": "when full, what was used longest ago is dropped first",
    "max_parallel": "answered at the same time (README: what it gains, measured)",
    "max_queue": "waiting for their turn; one more is refused at once",
    "max_wait_seconds": "a request that could not start by then is refused, and told why",
}


def _seed_text() -> str:
    home = _home_for_humans()
    return f"""\
# lmk configuration — these are the values in use. After a change: `lmk up` (it restarts the service).
# Every setting, with the tested models as lines to uncomment: config.yaml.example next to this file.

model:
  name: {DEFAULT_MODEL_NAME:<24} # a tested model; the list is in config.yaml.example
  # repo: mlx-community/Qwen3-30B-A3B-4bit   # {_WHAT["repo"]}
  # path: ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit   # {_WHAT["path"]}
  # reasoning_effort: low        # {_WHAT["reasoning_effort"]}
  # kv_cache_bits: 8             # {_WHAT["kv_cache_bits"]}
  # speculative_decoding: true   # {_WHAT["speculative_decoding"]}
  # thinking: false              # {_WHAT["thinking"]}
  # context_length: 65536        # {_WHAT["context_length"]}

listen:
  host: {DEFAULT_HOST}
  port: {DEFAULT_PORT}

cache:
  dir: {home}/cache              # {_WHAT["cache_dir"]}
  max_size: {DEFAULT_CACHE_MAX_SIZE}                 # {_WHAT["max_size"]}

requests:
  max_parallel: {DEFAULT_MAX_PARALLEL}                # {_WHAT["max_parallel"]}
  max_queue: {DEFAULT_MAX_QUEUE}                  # {_WHAT["max_queue"]}
  max_wait_seconds: {DEFAULT_MAX_WAIT_SECONDS}          # {_WHAT["max_wait_seconds"]}

log:
  dir: {home}/logs
"""


def _model_lines() -> str:
    """One commented `name:` line per tested model, smallest Mac first: choosing a model = uncommenting
    a line. Nothing else here — sizes, speeds and what each is good for are on the model pages."""
    order = sorted(TESTED_MODELS, key=lambda n: smallest_mac_gb(TESTED_MODELS[n]))
    return "\n".join(f"  # name: {name}" for name in order)


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
  name: {DEFAULT_MODEL_NAME:<24} # a tested model, one of these (each has a page under docs/models/):
{_model_lines()}
  # repo: mlx-community/Qwen3-30B-A3B-4bit   # {_WHAT["repo"]}
  # path: ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit   # {_WHAT["path"]}
  # The next three are the {DEFAULT_MODEL_NAME} page's recommendation; for another model its page says which to keep.
  reasoning_effort: low          # {_WHAT["reasoning_effort"]}
  # reasoning_effort: xhigh      # the template's own default; scored worse than low on every test
  kv_cache_bits: 8               # {_WHAT["kv_cache_bits"]}
  # kv_cache_bits: 16            # for a model whose page does not recommend 8
  speculative_decoding: true     # {_WHAT["speculative_decoding"]}
  # speculative_decoding: false  # for a model without a draft head: today all but {_draft_family()}
  # thinking: false              # {_WHAT["thinking"]}
  # context_length: 65536        # {_WHAT["context_length"]}
  # draft_tokens: 3              # {_WHAT["draft_tokens"]}

listen:
  host: {DEFAULT_HOST}
  port: {DEFAULT_PORT}
  # port: 8080

cache:
  dir: {home}/cache              # {_WHAT["cache_dir"]}
  max_size: {DEFAULT_CACHE_MAX_SIZE}                 # {_WHAT["max_size"]}
  # max_size: 50G

requests:
  max_parallel: {DEFAULT_MAX_PARALLEL}                # {_WHAT["max_parallel"]}
  # max_parallel: 1              # one at a time, each at full speed
  max_queue: {DEFAULT_MAX_QUEUE}                  # {_WHAT["max_queue"]}
  max_wait_seconds: {DEFAULT_MAX_WAIT_SECONDS}          # {_WHAT["max_wait_seconds"]}

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
