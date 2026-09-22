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
    "qwen3.8-27b-4bit": TestedModel(
        repo="lmstudio-community/Qwen3.8-27B-MLX-4bit", size_gb=16.1,
        note="Text and images in; tool calls and thinking. About 16 GB of memory for the weights. "
             "~39 tokens/s on an M3 Ultra (~33 on long agent conversations)."),
    "qwen3.8-27b-8bit": TestedModel(
        repo="lmstudio-community/Qwen3.8-27B-MLX-8bit", size_gb=29.5,
        note="The same model at 8-bit: less quantization loss, about 30 GB of memory for the weights, "
             "~23 tokens/s on an M3 Ultra. Prompt reading is as fast as 4-bit."),
}
DEFAULT_MODEL_NAME = "qwen3.8-27b-4bit"


def tested_models_markdown() -> str:
    """The table in README.md under "Models". A unit test holds the README to it."""
    rows = ["| `model.name` | HuggingFace repo | weights | notes |", "|---|---|---|---|"]
    for name, m in TESTED_MODELS.items():
        default = " (default)" if name == DEFAULT_MODEL_NAME else ""
        rows.append(f"| `{name}`{default} | [{m.repo}](https://huggingface.co/{m.repo}) | {m.size_gb:.1f} GB | {m.note} |")
    return "\n".join(rows)


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


def _hf_snapshot_dir(repo: str) -> Optional[Path]:
    """Reads the HF cache layout directly (models--org--name/refs/main names the commit
    under snapshots/). Not huggingface_hub.snapshot_download(local_files_only=True):
    importing mlx-engine replaces that function with one that always raises, so it
    would work or fail depending on what the process happened to import first."""
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_dir = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    try:
        commit = (repo_dir / "refs" / "main").read_text().strip()
    except OSError:
        return None
    snapshot = repo_dir / "snapshots" / commit
    return snapshot if snapshot.is_dir() else None


def resolve_model(source) -> ResolvedModel:
    """source: lmk.config.ModelSource. Never touches the network."""
    if source.kind == "path":
        if not source.path.is_dir():
            raise FileNotFoundError(f"model.path does not exist: {source.path}")
        return ResolvedModel(path=source.path, revision=None)

    snapshot = _hf_snapshot_dir(source.repo)
    if snapshot is None:
        raise ModelNotDownloaded(source.repo, "not downloaded")
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
