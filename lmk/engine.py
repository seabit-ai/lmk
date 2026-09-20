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
    context_length: int


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
    def chat_format(self) -> ChatFormat: ...
    def generate(self, prompt_text: str, *, max_tokens: Optional[int], request_id: str,
                 on_prefill: PrefillCallback) -> Generation: ...


class MlxEngine:
    """Loads the resident model at construction: there is no lazy / just-in-time
    loading in lmk (design §6.7)."""

    def __init__(self, model_id: str, model_path: Path, context_length: int):
        from mlx_engine.generate import load_model  # heavy import, kept out of module scope

        if not model_path.exists():
            raise FileNotFoundError(f"model path does not exist: {model_path}")
        from lmk.chatformat import TemplateChatFormat

        self._kit = load_model(model_path, max_kv_size=context_length, max_seq_nums=4)
        self._model = LoadedModel(id=model_id, path=model_path, context_length=context_length)
        self._format = TemplateChatFormat(self._kit.tokenizer)

    def loaded_model(self) -> LoadedModel:
        return self._model

    def chat_format(self) -> ChatFormat:
        return self._format

    def generate(self, prompt_text, *, max_tokens, request_id, on_prefill) -> Generation:
        from mlx_engine.generate import create_generator, tokenize
        from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter

        tokens = tokenize(self._kit, prompt_text)
        stats = GenerationStats(prompt_tokens=len(tokens))

        class Reporter(PromptProgressReporter):
            def begin(self, is_draft, cached_tokens, total_prompt_tokens, prefill_tokens_processed):
                stats.cached_tokens = cached_tokens
                return on_prefill(prefill_tokens_processed, total_prompt_tokens, cached_tokens)

            def update(self, is_draft, prefill_tokens_processed):
                return on_prefill(prefill_tokens_processed, stats.prompt_tokens, stats.cached_tokens)

            def finish(self, is_draft, prefill_tokens_processed=None):
                return True

        kwargs = {"prompt_progress_reporter": Reporter(), "request_id": request_id}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        def pieces():
            for result in create_generator(self._kit, tokens, **kwargs):
                stats.completion_tokens += len(result.tokens)
                if result.text:
                    yield result.text

        return Generation(pieces=pieces(), stats=stats)


class FakeEngine:
    """Scripted engine for unit tests: replays text pieces and reports the given stats."""

    def __init__(self, model: LoadedModel, chat_format: Optional[ChatFormat] = None,
                 script: Optional[list[str]] = None, stats: Optional[GenerationStats] = None,
                 prefill_steps: Optional[list[int]] = None):
        self._model = model
        self._format = chat_format
        self._script = script or []
        self._stats = stats or GenerationStats()
        self._prefill_steps = prefill_steps or []
        self.requests: list[dict] = []

    def loaded_model(self) -> LoadedModel:
        return self._model

    def chat_format(self) -> ChatFormat:
        return self._format

    def generate(self, prompt_text, *, max_tokens, request_id, on_prefill) -> Generation:
        self.requests.append({"prompt": prompt_text, "max_tokens": max_tokens, "request_id": request_id})
        stats = GenerationStats(**vars(self._stats))

        def pieces():
            on_prefill(0, stats.prompt_tokens, stats.cached_tokens)
            for processed in self._prefill_steps:
                if not on_prefill(processed, stats.prompt_tokens, stats.cached_tokens):
                    return
            yield from self._script

        return Generation(pieces=pieces(), stats=stats)
