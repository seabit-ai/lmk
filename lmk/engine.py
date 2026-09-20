"""The one seam between lmk and mlx-engine (design §1: keep the engine-facing
part behind a narrow interface). Everything above this file is testable with
FakeEngine — no GPU, no model."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

from lmk.chatformat import ChatFormat


@dataclass(frozen=True)
class LoadedModel:
    id: str
    path: Path
    context_length: int  # the value in use — the engine may fit it to the memory it finds
    requested_context_length: Optional[int] = None


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0


# on_prefill(processed, total, cached) -> keep going?  False cancels during prefill.
PrefillCallback = Callable[[int, int, int], bool]


@dataclass
class Generation:
    """One streamed generation: iterate for text pieces; stats fill in as it runs."""
    pieces: Iterator[str]
    stats: GenerationStats = field(default_factory=GenerationStats)

    def __iter__(self):
        return self.pieces


class Engine(Protocol):
    def loaded_model(self) -> LoadedModel: ...
    def input_modalities(self) -> list[str]: ...
    def chat_format(self) -> ChatFormat: ...
    def cache_stats(self) -> Optional[dict]: ...
    def generate(self, prompt_text: str, *, max_tokens: Optional[int], request_id: str,
                 on_prefill: PrefillCallback, images_b64: Optional[list[str]] = None) -> Generation: ...


class MlxEngine:
    """Loads the resident model at construction: there is no lazy / just-in-time
    loading in lmk (design §6.7)."""

    def __init__(self, model_id: str, model_path: Path, context_length: Optional[int] = None, *,
                 cache_dir: Optional[Path] = None, cache_max_bytes: int = 0,
                 repo: Optional[str] = None, revision: Optional[str] = None):
        from mlx_engine.generate import get_runtime_load_info, load_model  # heavy import, kept out of module scope

        if not model_path.exists():
            raise FileNotFoundError(f"model path does not exist: {model_path}")
        from lmk.chatformat import TemplateChatFormat
        from lmk.models import native_context_length

        requested = context_length or native_context_length(model_path)
        if requested is None:
            raise ValueError(f"{model_path}/config.json does not say how long the model's context is — "
                             "set model.context_length")
        self._cache_stores: list = []
        if cache_dir is not None:
            _install_persistent_cache(cache_dir, cache_max_bytes, model_path, repo, revision, self._cache_stores)
        self._kit = load_model(model_path, max_kv_size=requested, max_seq_nums=4)
        in_use = get_runtime_load_info(self._kit).get("context_length") or requested
        self._model = LoadedModel(id=model_id, path=model_path, context_length=in_use,
                                  requested_context_length=requested)
        self._format = TemplateChatFormat(self._kit.tokenizer)

    def loaded_model(self) -> LoadedModel:
        return self._model

    def chat_format(self) -> ChatFormat:
        return self._format

    def input_modalities(self) -> list[str]:
        # the engine picks its vision kit for models whose config has vision_config
        return ["text", "image"] if "Vision" in type(self._kit).__name__ else ["text"]

    def cache_stats(self) -> Optional[dict]:
        return self._cache_stores[-1].stats() if self._cache_stores else None

    def close(self) -> None:
        """Drains the engine's cache I/O thread: records still queued for disk
        are written before the process goes away."""
        from mlx_engine.generate import unload

        unload(self._kit)

    def generate(self, prompt_text, *, max_tokens, request_id, on_prefill, images_b64=None) -> Generation:
        from mlx_engine.generate import create_generator, tokenize
        from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter

        tokens = tokenize(self._kit, prompt_text)
        stats = GenerationStats(prompt_tokens=len(tokens))

        class Reporter(PromptProgressReporter):
            def begin(self, is_draft, cached_tokens, total_prompt_tokens, prefill_tokens_processed):
                stats.cached_tokens = cached_tokens
                # with images the text-only count misses the expanded vision tokens;
                # the engine's own total is the one that was actually prefilled
                stats.prompt_tokens = max(stats.prompt_tokens, total_prompt_tokens)
                return on_prefill(prefill_tokens_processed, total_prompt_tokens, cached_tokens)

            def update(self, is_draft, prefill_tokens_processed):
                return on_prefill(prefill_tokens_processed, stats.prompt_tokens, stats.cached_tokens)

            def finish(self, is_draft, prefill_tokens_processed=None):
                return True

        kwargs = {"prompt_progress_reporter": Reporter(), "request_id": request_id}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if images_b64:
            kwargs["images_b64"] = images_b64

        def pieces():
            for result in create_generator(self._kit, tokens, **kwargs):
                stats.completion_tokens += len(result.tokens)
                if result.text:
                    yield result.text

        return Generation(pieces=pieces(), stats=stats)


def engine_commit() -> str:
    return (Path(__file__).resolve().parent.parent / "ENGINE_COMMIT").read_text().strip()


def _install_persistent_cache(cache_dir: Path, max_bytes: int, model_path: Path, repo: Optional[str],
                              revision: Optional[str], created: list) -> None:
    """The engine constructs its cache store itself (model_kit.py), with the
    directory hard-coded; the one way in is to swap the class it names."""
    import mlx_engine.model_kit.batched_vision.model_kit as vision_kit

    from lmk.persistcache import make_persistent_store_class, model_identity, prepare_cache_root

    identity = model_identity(model_path, repo=repo, revision=revision)
    live_budget = prepare_cache_root(cache_dir, identity, max_bytes)
    vision_kit.VlmPromptCacheStore = make_persistent_store_class(cache_dir / identity, live_budget,
                                                                 engine_commit(), created)


class FakeEngine:
    """Scripted engine for unit tests: replays text pieces and reports the given stats."""

    def __init__(self, model: LoadedModel, chat_format: Optional[ChatFormat] = None,
                 script: Optional[list[str]] = None, stats: Optional[GenerationStats] = None,
                 prefill_steps: Optional[list[int]] = None, modalities: Optional[list[str]] = None,
                 cache: Optional[dict] = None):
        self._model = model
        self._format = chat_format
        self._script = script or []
        self._stats = stats or GenerationStats()
        self._prefill_steps = prefill_steps or []
        self._modalities = modalities or ["text"]
        self._cache = cache
        self.requests: list[dict] = []

    def cache_stats(self) -> Optional[dict]:
        return self._cache

    def loaded_model(self) -> LoadedModel:
        return self._model

    def chat_format(self) -> ChatFormat:
        return self._format

    def input_modalities(self) -> list[str]:
        return self._modalities

    def generate(self, prompt_text, *, max_tokens, request_id, on_prefill, images_b64=None) -> Generation:
        self.requests.append({"prompt": prompt_text, "max_tokens": max_tokens, "request_id": request_id,
                              "images_b64": images_b64})
        stats = GenerationStats(**vars(self._stats))

        def pieces():
            on_prefill(0, stats.prompt_tokens, stats.cached_tokens)
            for processed in self._prefill_steps:
                if not on_prefill(processed, stats.prompt_tokens, stats.cached_tokens):
                    return
            yield from self._script

        return Generation(pieces=pieces(), stats=stats)
