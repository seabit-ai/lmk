"""~/.kitten/lmk.yaml — machine-level config (design §7)."""
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".kitten" / "lmk.yaml"


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class ModelConfig:
    id: str  # the name clients put in "model"
    path: Path
    context_length: int


@dataclass(frozen=True)
class LmkConfig:
    model: ModelConfig
    host: str
    port: int
    cache_dir: Path
    log_dir: Path


def _require(section: dict, key: str, where: str):
    if key not in section or section[key] in (None, ""):
        raise ConfigError(f"{where}.{key} is required")
    return section[key]


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> LmkConfig:
    if not path.exists():
        raise ConfigError(f"no config at {path} — see lmk/README.md for a starting point")
    raw = yaml.safe_load(path.read_text()) or {}
    model = raw.get("model") or {}
    listen = raw.get("listen") or {}
    model_path = Path(str(_require(model, "path", "model"))).expanduser()
    context_length = int(_require(model, "context_length", "model"))
    if context_length <= 0:
        raise ConfigError("model.context_length must be positive")
    return LmkConfig(
        model=ModelConfig(
            id=str(model.get("id") or model_path.name.lower()),
            path=model_path,
            context_length=context_length,
        ),
        # No default port: two servers guessing the same number is a silent conflict.
        host=str(listen.get("host") or "127.0.0.1"),
        port=int(_require(listen, "port", "listen")),
        cache_dir=Path(str((raw.get("cache") or {}).get("dir") or "~/.kitten/lmk/cache")).expanduser(),
        log_dir=Path(str((raw.get("log") or {}).get("dir") or "~/Library/Logs/kitten")).expanduser(),
    )
