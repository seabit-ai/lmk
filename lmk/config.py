"""~/.lmk/config.yaml — every key is optional (design OOBE §C).

Defaults live here, in code. The seeded config.yaml writes every value in use (OOBE C4) —
copied from these defaults once, then the file is the user's; a missing key still means
the default here.
"""
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from lmk.models import DEFAULT_MODEL_NAME, TESTED_MODELS

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 1235
DEFAULT_CACHE_MAX_SIZE = "200G"
# Why these numbers: docs/design/2026-09-20-memory-guard.md §F.
# What running side by side gains depends on how long the conversations are (research MG-006, MG-008):
# two tiny prompts together reach 1.7x the speed of one; two 27k-token conversations reach 1.0x.
DEFAULT_MAX_PARALLEL = 2
DEFAULT_MAX_QUEUE = 16           # six sessions, each with a turn and a background call, and room to spare
DEFAULT_MAX_WAIT_SECONDS = 600   # behind one 100k-token cold prompt (~5 min) plus one 8k-token answer (~4 min)


def lmk_home() -> Path:
    return Path(os.environ.get("LMK_HOME") or Path.home() / ".lmk")


def config_path() -> Path:
    return lmk_home() / "config.yaml"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class ModelSource:
    """Exactly one way of naming the model: a tested name, any HF repo, or a local directory."""
    kind: str  # "name" | "repo" | "path"
    value: str

    @property
    def repo(self) -> Optional[str]:
        if self.kind == "name":
            return TESTED_MODELS[self.value].repo
        return self.value if self.kind == "repo" else None

    @property
    def path(self) -> Optional[Path]:
        return Path(self.value).expanduser() if self.kind == "path" else None


KV_CACHE_BITS = (16, 8, 4)  # what the engine's batched path quantizes to; 16 means no quantization
MODEL_KEYS = {"name", "repo", "path", "context_length", "thinking", "reasoning_effort", "kv_cache_bits",
              "speculative_decoding", "draft", "draft_tokens", "id"}  # id: refused with its own message below
DRAFT_KINDS = ("mtp", "dflash2")


@dataclass(frozen=True)
class ModelConfig:
    id: str  # what clients put in "model": the tested name, or the repo / directory name in lower case
    source: ModelSource
    context_length: Optional[int]  # None: the model's own maximum
    thinking: Optional[bool]       # None: the template's default (Qwen: on). False: the model answers without thinking
    reasoning_effort: Optional[str]  # for templates that know it (Qwen3.8: low / medium / xhigh); None: the template's default
    kv_cache_bits: int = 16        # 16 = the model's own precision; 8 / 4 quantize the KV cache (more context, same memory)
    speculative_decoding: bool = False  # draft tokens and let the model check them; needs the draft lmk up fetched
    draft: Optional[str] = None         # which drafter: mtp (the model's own head) / dflash2; None: the model page's default
    draft_tokens: Optional[int] = None  # tokens drafted per round; None: the draft model's own setting

    def template_kwargs(self) -> dict:
        kwargs = {}
        if self.thinking is not None:
            kwargs["enable_thinking"] = self.thinking
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs


@dataclass(frozen=True)
class RequestsConfig:
    max_parallel: int       # answered at the same time
    max_queue: int          # waiting for their turn; one more is refused at once
    max_wait_seconds: int   # a request that could not start by then is refused


@dataclass(frozen=True)
class LmkConfig:
    model: ModelConfig
    host: str
    port: int
    cache_dir: Path
    cache_max_bytes: int
    log_dir: Path
    requests: RequestsConfig = RequestsConfig(DEFAULT_MAX_PARALLEL, DEFAULT_MAX_QUEUE, DEFAULT_MAX_WAIT_SECONDS)


_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT])?I?B?\s*$", re.IGNORECASE)
_UNIT = {None: 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_size(text) -> int:
    m = _SIZE.match(str(text))
    if not m:
        raise ConfigError(f"cannot read size {text!r} — write it like 200G or 500M")
    return int(float(m.group(1)) * _UNIT[m.group(2).upper() if m.group(2) else None])


def _section(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a section with keys under it, not {value!r}")
    return value


def _positive_int(section: dict, where: str, key: str, default: int) -> int:
    value = section.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{where}.{key} must be a whole number, 1 or more (found {value!r})")
    return value


def _model_source(model: dict) -> ModelSource:
    unknown = sorted(set(model) - MODEL_KEYS)
    if unknown:
        raise ConfigError(f"model: unknown setting {', '.join(unknown)} — the settings are {', '.join(sorted(MODEL_KEYS))} "
                          "(a misspelling would otherwise be ignored silently)")
    named = [k for k in ("name", "repo", "path") if model.get(k)]
    if len(named) > 1:
        raise ConfigError(f"model: name, repo and path each say which model to load, so keep only one of them (found {', '.join(named)})")
    if not named:
        return ModelSource("name", DEFAULT_MODEL_NAME)
    kind, value = named[0], str(model[named[0]])
    if kind == "name" and value not in TESTED_MODELS:
        raise ConfigError(f"model.name {value!r} is not in the tested list ({', '.join(TESTED_MODELS)}) — "
                          "for any other HuggingFace model use model.repo")
    return ModelSource(kind, value)


def _default_model_id(source: ModelSource) -> str:
    if source.kind == "name":
        return source.value
    return Path(source.value).name.lower()


def load_config(path: Optional[Path] = None) -> LmkConfig:
    path = path or config_path()
    raw = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must be a YAML mapping")
    model, listen, cache, log, requests = (_section(raw, k) for k in ("model", "listen", "cache", "log", "requests"))
    source = _model_source(model)
    context_length = model.get("context_length")
    if context_length is not None:
        context_length = int(context_length)
        if context_length <= 0:
            raise ConfigError("model.context_length must be positive")
    if "id" in model:
        raise ConfigError("model.id is gone (2026-09-22): the name clients use is always the model's name — "
                          f"here that is {_default_model_id(source)!r}. Remove the id: line")
    thinking = model.get("thinking")
    if thinking is not None and not isinstance(thinking, bool):
        raise ConfigError("model.thinking must be true or false")
    effort = model.get("reasoning_effort")
    if effort is not None and (not isinstance(effort, str) or not effort):
        raise ConfigError("model.reasoning_effort must be a word the model's chat template knows, such as low / medium / xhigh")
    kv_bits = model.get("kv_cache_bits", 16)
    if isinstance(kv_bits, bool) or not isinstance(kv_bits, int) or kv_bits not in KV_CACHE_BITS:
        raise ConfigError(f"model.kv_cache_bits must be one of {', '.join(map(str, KV_CACHE_BITS))} "
                          f"(16 = the model's own precision; 8 halves what each token of context costs in memory) — got {kv_bits!r}")
    speculative = model.get("speculative_decoding", False)
    if not isinstance(speculative, bool):
        raise ConfigError("model.speculative_decoding must be true or false")
    draft = model.get("draft")
    if draft is not None and draft not in DRAFT_KINDS:
        raise ConfigError(f"model.draft must be one of {', '.join(DRAFT_KINDS)} (which drafter speculative decoding uses; "
                          f"leave it out for the model page's default) — got {draft!r}")
    draft_tokens = model.get("draft_tokens")
    if draft_tokens is not None and (isinstance(draft_tokens, bool) or not isinstance(draft_tokens, int)
                                     or not 1 <= draft_tokens <= 16):
        raise ConfigError(f"model.draft_tokens must be a whole number from 1 to 16 (tokens drafted per round) — got {draft_tokens!r}")
    return LmkConfig(
        model=ModelConfig(id=_default_model_id(source), source=source,
                          context_length=context_length, thinking=thinking, reasoning_effort=effort,
                          kv_cache_bits=int(kv_bits), speculative_decoding=speculative, draft=draft, draft_tokens=draft_tokens),
        host=str(listen.get("host") or DEFAULT_HOST),
        port=int(listen.get("port") or DEFAULT_PORT),
        cache_dir=Path(str(cache.get("dir") or lmk_home() / "cache")).expanduser(),
        cache_max_bytes=parse_size(cache.get("max_size") or DEFAULT_CACHE_MAX_SIZE),
        log_dir=Path(str(log.get("dir") or lmk_home() / "logs")).expanduser(),
        requests=RequestsConfig(
            max_parallel=_positive_int(requests, "requests", "max_parallel", DEFAULT_MAX_PARALLEL),
            max_queue=_positive_int(requests, "requests", "max_queue", DEFAULT_MAX_QUEUE),
            max_wait_seconds=_positive_int(requests, "requests", "max_wait_seconds", DEFAULT_MAX_WAIT_SECONDS)),
    )


def app_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def build_id() -> str:
    """Written by the installer. A running service reports the build it was started
    from, so `lmk up` can tell that the code on disk is newer."""
    try:
        return (app_dir() / "BUILD").read_text().strip() or "dev"
    except OSError:
        return "dev"


def fingerprint(cfg: LmkConfig, revision: Optional[str]) -> str:
    """Everything a running service was started with. `lmk up` compares the running
    one's with what the files on disk say now; different means restart."""
    return hashlib.sha256(f"{build_id()}|{cfg!r}|{revision}".encode()).hexdigest()[:16]
