"""The one seam between lmk and mlx-engine (design §1: keep the engine-facing
part behind a narrow interface). Everything above this file is testable with
FakeEngine — no GPU, no model."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class LoadedModel:
    id: str
    path: Path
    context_length: int


class Engine(Protocol):
    def loaded_model(self) -> LoadedModel: ...


class MlxEngine:
    """Loads the resident model at construction: there is no lazy / just-in-time
    loading in lmk (design §6.7)."""

    def __init__(self, model_id: str, model_path: Path, context_length: int):
        from mlx_engine.generate import load_model  # heavy import, kept out of module scope

        if not model_path.exists():
            raise FileNotFoundError(f"model path does not exist: {model_path}")
        self._kit = load_model(model_path, max_kv_size=context_length, max_seq_nums=4)
        self._model = LoadedModel(id=model_id, path=model_path, context_length=context_length)

    def loaded_model(self) -> LoadedModel:
        return self._model


class FakeEngine:
    def __init__(self, model: LoadedModel):
        self._model = model

    def loaded_model(self) -> LoadedModel:
        return self._model
