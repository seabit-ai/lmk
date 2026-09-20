"""The tested-model list, and finding a model on disk.

lmk owns no model directory (design OOBE §B): HuggingFace models live in the
shared HF cache, where every other tool can see them too. Downloading is its
own explicit command (`lmk pull`), never a side effect of starting.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TestedModel:
    __test__ = False  # not a pytest class

    repo: str
    size_gb: float
    note: str


# Models we have run end to end with a real agent. config.yaml.example prints this
# list, so `note` is written for someone choosing a model, not for us.
TESTED_MODELS: dict[str, TestedModel] = {
    "qwen3.8-27b": TestedModel(
        repo="lmstudio-community/Qwen3.8-27B-MLX-4bit", size_gb=16.1,
        note="Text and images in; tool calls and thinking. About 16 GB of memory for the weights. ~33 tokens/s on an M3 Ultra."),
}
DEFAULT_MODEL_NAME = "qwen3.8-27b"


class ModelNotDownloaded(Exception):
    def __init__(self, repo: str, why: str):
        super().__init__(f"{repo}: {why}")
        self.repo = repo
        self.why = why


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    revision: Optional[str]  # the HF commit hash; None for a local directory


def missing_weight_files(model_dir: Path) -> list[str]:
    """A snapshot directory appears as soon as a download starts; the weight
    index says which shards must be there before the model can load."""
    index = model_dir / "model.safetensors.index.json"
    if not index.exists():
        return [] if any(model_dir.glob("*.safetensors")) else ["*.safetensors"]
    shards = set(json.loads(index.read_text())["weight_map"].values())
    return sorted(s for s in shards if not (model_dir / s).exists())


def resolve_model(source) -> ResolvedModel:
    """source: lmk.config.ModelSource. Never touches the network."""
    if source.kind == "path":
        if not source.path.is_dir():
            raise FileNotFoundError(f"model.path does not exist: {source.path}")
        return ResolvedModel(path=source.path, revision=None)

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot = Path(snapshot_download(source.repo, local_files_only=True))
    except LocalEntryNotFoundError:
        raise ModelNotDownloaded(source.repo, "not downloaded") from None
    missing = missing_weight_files(snapshot)
    if missing:
        raise ModelNotDownloaded(source.repo, f"download incomplete ({len(missing)} weight file(s) missing)")
    return ResolvedModel(path=snapshot, revision=snapshot.name)


def native_context_length(model_dir: Path) -> Optional[int]:
    """The model's own maximum, from its config.json (VLMs keep it under text_config)."""
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    for holder in (cfg.get("text_config") or {}, cfg):
        value = holder.get("max_position_embeddings")
        if isinstance(value, int) and value > 0:
            return value
    return None
