"""Sampling parameters: OpenAI's names in, the engine's names out.

The engine's own default is greedy (temp 0, sampling.py in mlx-engine) and it
never reads the model's generation_config.json; lmk does, so that with no
parameters in the request the model runs the way its authors shipped it
(THK-001, research/2026-09-20-thinking-length)."""
import json
from pathlib import Path
from typing import Optional


class SamplingError(ValueError):
    def __init__(self, param: str, message: str):
        super().__init__(f"{param} must be {message}")
        self.param = param


# generation_config.json key -> engine kwarg
_MODEL_CONFIG_KEYS = {"temperature": "temp", "top_p": "top_p", "top_k": "top_k"}

# OpenAI request field -> engine kwarg. `stop` is handled apart (list shape); `seed` is
# accepted by the engine's signature but ignored on the batched path lmk runs on
# (generate.py: "Seed arg is ignored for batched gen"), so it is reported, not passed.
_REQUEST_KEYS = {"temperature": "temp", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
                 "repetition_penalty": "repetition_penalty"}
IGNORED_KEYS = ("seed",)
MAX_STOP_STRINGS = 4  # OpenAI's limit; the engine has none


def model_defaults(model_path: Path) -> dict:
    """What the model's authors shipped in generation_config.json, as engine kwargs.
    do_sample: false means greedy; a missing or unreadable file means "no opinion"."""
    try:
        cfg = json.loads((model_path / "generation_config.json").read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(cfg, dict):
        return {}
    if cfg.get("do_sample") is False:
        return {"temp": 0.0}
    return {engine_key: cfg[key] for key, engine_key in _MODEL_CONFIG_KEYS.items()
            if isinstance(cfg.get(key), (int, float)) and not isinstance(cfg.get(key), bool)}


def parse_sampling(body: dict, defaults: dict) -> tuple[dict, list[str]]:
    """Returns (engine kwargs, request fields that were understood but ignored).
    Fields absent or null fall back to `defaults`; unknown fields are ignored silently, as OpenAI does."""
    sampling = dict(defaults)
    for field, engine_key in _REQUEST_KEYS.items():
        value = body.get(field)
        if value is None:
            continue
        sampling[engine_key] = _checked_number(field, value)
    stop = body.get("stop")
    if stop is not None:
        sampling["stop_strings"] = _checked_stop(stop)
    ignored = [k for k in IGNORED_KEYS if body.get(k) is not None]
    return sampling, ignored


def _checked_number(field: str, value):
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if field == "top_k":
        if not (isinstance(value, int) and not isinstance(value, bool)) or value < 0:
            raise SamplingError(field, "an integer of 0 or more")
        return value
    if not is_number:
        raise SamplingError(field, "a number")
    if field == "temperature" and not 0 <= value <= 2:
        raise SamplingError(field, "between 0 and 2")
    if field == "top_p" and not 0 < value <= 1:
        raise SamplingError(field, "between 0 (exclusive) and 1")
    if field == "min_p" and not 0 <= value <= 1:
        raise SamplingError(field, "between 0 and 1")
    if field == "repetition_penalty" and not value > 0:
        raise SamplingError(field, "greater than 0")
    return float(value)


def _checked_stop(stop) -> list[str]:
    items = [stop] if isinstance(stop, str) else stop
    if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
        raise SamplingError("stop", "a string or a list of strings")
    if len(items) > MAX_STOP_STRINGS:
        raise SamplingError("stop", f"at most {MAX_STOP_STRINGS} strings")
    if any(s == "" for s in items):
        raise SamplingError("stop", "non-empty strings")
    return items


def describe(sampling: Optional[dict]) -> str:
    """One line for humans: `temp 1.0 · top_p 0.95 · top_k 20`; `greedy` when temp is 0."""
    if not sampling:
        return "engine default (greedy)"
    if sampling.get("temp", None) == 0:
        return "greedy (temp 0)"
    return " · ".join(f"{k} {v}" for k, v in sampling.items() if k != "stop_strings")
