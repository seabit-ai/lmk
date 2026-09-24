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
    kv_cache_bits: int = 16
    speculative_decoding: bool = False
    draft_kind: Optional[str] = None   # mtp / dflash2 when a draft is loaded


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    draft_accepted: Optional[int] = None  # speculative decoding: drafted tokens the model agreed with ...
    draft_drafted: Optional[int] = None   # ... out of how many it drafted (shared when requests overlap)


@dataclass
class Preflight:
    """What can be known about a request before it is admitted (design memory-guard §B2, §C)."""
    prompt_tokens: int
    uncached_tokens: Optional[int]   # None: could not be told (images, or no index yet)
    tokens: Optional[list] = None    # handed back to generate() so the prompt is tokenized once


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
    def draft_stats(self) -> Optional[dict]:
        """Speculative decoding since start: rounds, accepted, drafted. None without a draft model."""
        return None
    def sampling_defaults(self) -> dict:
        """Engine kwargs used when a request names no sampling parameter (see lmk.sampling)."""
        ...
    def thinking_enabled(self) -> bool:
        """What every prompt says about thinking: the configured constant, else the family's default."""
        ...
    def reasoning_effort(self) -> Optional[str]:
        """The configured model.reasoning_effort, None when the template's own default applies."""
        ...
    def preflight(self, prompt_text: str, images_b64: Optional[list[str]] = None) -> Preflight: ...
    def token_budget(self) -> Optional[int]: ...
    def gpu_memory_bytes(self) -> int: ...
    def gpu_memory_peak_bytes(self) -> int: ...
    def generate(self, prompt_text: str, *, max_tokens: Optional[int], request_id: str,
                 on_prefill: PrefillCallback, images_b64: Optional[list[str]] = None,
                 sampling: Optional[dict] = None, tokens: Optional[list] = None) -> Generation: ...


def runtime_import_error() -> Optional[str]:
    """Why the model runtime cannot be imported, or None. Importing it is a side effect
    (it replaces huggingface_hub.snapshot_download) — only `lmk serve` may call this."""
    try:
        import mlx_engine.generate  # noqa: F401 - the probe is the import
    except ImportError as e:
        return str(e)
    return None


class MlxEngine:
    """Loads the resident model at construction: there is no lazy / just-in-time
    loading in lmk (design §6.7)."""

    def __init__(self, model_id: str, model_path: Path, context_length: Optional[int] = None, *,
                 cache_dir: Optional[Path] = None, cache_max_bytes: Optional[int] = None,
                 repo: Optional[str] = None, revision: Optional[str] = None, max_parallel: int = 2,
                 template_kwargs: Optional[dict] = None, kv_cache_bits: int = 16,
                 draft_path: Optional[Path] = None, draft_kind: Optional[str] = None, draft_tokens: Optional[int] = None):
        from mlx_engine.generate import get_runtime_load_info, load_draft_model, load_model  # heavy import, kept out of module scope

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
            _install_persistent_cache(cache_dir, cache_max_bytes, model_path, repo, revision, self._cache_stores,
                                      kv_cache_bits=kv_cache_bits)
        # lmk's own queue enforces max_parallel where people can see who waits and why;
        # the engine gets the same number as a backstop
        self._kit = load_model(model_path, max_kv_size=requested, max_seq_nums=max_parallel,
                               kv_bits=None if kv_cache_bits == 16 else kv_cache_bits)
        self._draft_tokens = draft_tokens
        if draft_path is not None:
            load_draft_model(self._kit, str(draft_path))
        in_use = get_runtime_load_info(self._kit).get("context_length") or requested
        self._model = LoadedModel(id=model_id, path=model_path, context_length=in_use,
                                  requested_context_length=requested, kv_cache_bits=kv_cache_bits,
                                  speculative_decoding=draft_path is not None,
                                  draft_kind=draft_kind if draft_path is not None else None)
        self._format = TemplateChatFormat(self._kit.tokenizer, template_kwargs)
        self._thinking = bool((template_kwargs or {}).get("enable_thinking", self._format.dialect.thinking_default))
        self._effort = (template_kwargs or {}).get("reasoning_effort")
        from lmk.sampling import model_defaults
        self._sampling_defaults = model_defaults(model_path)

    def loaded_model(self) -> LoadedModel:
        return self._model

    def sampling_defaults(self) -> dict:
        return dict(self._sampling_defaults)

    def thinking_enabled(self) -> bool:
        return self._thinking

    def reasoning_effort(self) -> Optional[str]:
        return self._effort

    def chat_format(self) -> ChatFormat:
        return self._format

    def input_modalities(self) -> list[str]:
        # the engine picks its vision kit for models whose config has vision_config
        return ["text", "image"] if "Vision" in type(self._kit).__name__ else ["text"]

    def cache_stats(self) -> Optional[dict]:
        return self._cache_stores[-1].stats() if self._cache_stores else None

    def _drafter_counters(self) -> Optional[tuple[int, int, int]]:
        drafter = getattr(self._kit, "_drafter", None)
        if drafter is None:
            return None
        return drafter.rounds, drafter.accepted, drafter.drafted

    def draft_stats(self) -> Optional[dict]:
        counters = self._drafter_counters()
        if counters is None:
            return None
        rounds, accepted, drafted = counters
        return {"rounds": rounds, "accepted": accepted, "drafted": drafted}

    def preflight(self, prompt_text, images_b64=None) -> Preflight:
        from mlx_engine.generate import tokenize

        tokens = tokenize(self._kit, prompt_text)
        uncached = None
        store = getattr(self._kit, "_prompt_cache_store", None)
        if not images_b64 and store is not None:  # image spans are part of the cache key; not reproduced here
            try:
                # reads the store's in-memory index only. The index belongs to the engine's cache
                # I/O thread; a read that collides with a write is answered "unknown", which the
                # queue treats as a long prompt — the cautious side.
                plan = store.plan_longest_prefix_restore(tokens, [])
                uncached = len(tokens) - (plan.cached_prefix_len if plan is not None else 0)
            except Exception as e:  # noqa: BLE001
                from lmk import log

                log.warn("LmkPreflightLookupFailed", "could not tell how much of the prompt is cached", error=repr(e))
        return Preflight(prompt_tokens=len(tokens), uncached_tokens=uncached, tokens=tokens)

    def token_budget(self) -> Optional[int]:
        """How many tokens of KV cache fit this Mac next to the weights, from the coefficients
        the engine measured at load (context_fit.py). None when it made no fit."""
        fit = getattr(self._kit, "_context_fit_result", None)
        per_token = getattr(getattr(fit, "profile", None), "full_kv_bytes_per_token", 0)
        if not per_token:
            return None
        return max(0, (fit.safe_ceiling_bytes - fit.baseline_bytes) // per_token)

    def gpu_memory_bytes(self) -> int:
        import mlx.core as mx

        return int(mx.get_active_memory() + mx.get_cache_memory())

    def gpu_memory_peak_bytes(self) -> int:
        import mlx.core as mx

        return int(mx.get_peak_memory())

    def close(self) -> None:
        """Drains the engine's cache I/O thread: records still queued for disk
        are written before the process goes away."""
        from mlx_engine.generate import unload

        unload(self._kit)

    def generate(self, prompt_text, *, max_tokens, request_id, on_prefill, images_b64=None, sampling=None,
                 tokens=None) -> Generation:
        from mlx_engine.generate import create_generator, tokenize
        from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter

        if tokens is None:
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
        kwargs.update(sampling or {})  # the engine's own names: temp, top_p, top_k, seed
        if self._draft_tokens and self._drafter_counters() is not None:
            kwargs["num_draft_tokens"] = self._draft_tokens

        def pieces():
            before = self._drafter_counters()
            for result in create_generator(self._kit, tokens, **kwargs):
                stats.completion_tokens += len(result.tokens)
                if result.text:
                    yield result.text
            after = self._drafter_counters()
            if before is not None and after is not None:
                stats.draft_accepted = after[1] - before[1]
                stats.draft_drafted = after[2] - before[2]

        return Generation(pieces=pieces(), stats=stats)


def engine_commit() -> str:
    return (Path(__file__).resolve().parent.parent / "ENGINE_COMMIT").read_text().strip()


def _install_persistent_cache(cache_dir: Path, max_bytes: Optional[int], model_path: Path, repo: Optional[str],
                              revision: Optional[str], created: list, kv_cache_bits: int = 16) -> None:
    """The engine constructs its cache store itself (model_kit.py), with the
    directory hard-coded; the one way in is to swap the class it names."""
    import mlx_engine.model_kit.batched_vision.model_kit as vision_kit

    from lmk.persistcache import make_persistent_store_class, model_identity, prepare_cache_root

    identity = model_identity(model_path, repo=repo, revision=revision, kv_cache_bits=kv_cache_bits)
    if max_bytes is None:
        max_bytes = 1 << 62  # no limit of ours; the engine's own budget still applies
    live_budget = prepare_cache_root(cache_dir, identity, max_bytes)
    vision_kit.VlmPromptCacheStore = make_persistent_store_class(cache_dir / identity, live_budget,
                                                                 engine_commit(), created)


class FakeEngine:
    """Scripted engine for unit tests: replays text pieces and reports the given stats."""

    def __init__(self, model: LoadedModel, chat_format: Optional[ChatFormat] = None,
                 script: Optional[list[str]] = None, stats: Optional[GenerationStats] = None,
                 prefill_steps: Optional[list[int]] = None, modalities: Optional[list[str]] = None,
                 cache: Optional[dict] = None, token_budget: Optional[int] = None, gpu_bytes: int = 0,
                 sampling_defaults: Optional[dict] = None, thinking: bool = True,
                 reasoning_effort: Optional[str] = None, draft: Optional[dict] = None):
        self._model = model
        self._sampling_defaults = sampling_defaults or {}
        self._thinking = thinking
        self._effort = reasoning_effort
        self._draft = draft
        self._format = chat_format
        self._script = script or []
        self._stats = stats or GenerationStats()
        self._prefill_steps = prefill_steps or []
        self._modalities = modalities or ["text"]
        self._cache = cache
        self._token_budget = token_budget
        self._gpu_bytes = gpu_bytes
        self.requests: list[dict] = []

    def preflight(self, prompt_text, images_b64=None) -> Preflight:
        uncached = None if images_b64 else self._stats.prompt_tokens - self._stats.cached_tokens
        return Preflight(prompt_tokens=self._stats.prompt_tokens, uncached_tokens=uncached)

    def token_budget(self) -> Optional[int]:
        return self._token_budget

    def gpu_memory_bytes(self) -> int:
        return self._gpu_bytes

    def gpu_memory_peak_bytes(self) -> int:
        return self._gpu_bytes

    def cache_stats(self) -> Optional[dict]:
        return self._cache

    def loaded_model(self) -> LoadedModel:
        return self._model

    def sampling_defaults(self) -> dict:
        return dict(self._sampling_defaults)

    def thinking_enabled(self) -> bool:
        return self._thinking

    def reasoning_effort(self) -> Optional[str]:
        return self._effort

    def draft_stats(self) -> Optional[dict]:
        return self._draft

    def chat_format(self) -> ChatFormat:
        return self._format

    def input_modalities(self) -> list[str]:
        return self._modalities

    def generate(self, prompt_text, *, max_tokens, request_id, on_prefill, images_b64=None, sampling=None,
                 tokens=None) -> Generation:
        self.requests.append({"prompt": prompt_text, "max_tokens": max_tokens, "request_id": request_id,
                              "images_b64": images_b64, "sampling": sampling})
        stats = GenerationStats(**vars(self._stats))

        def pieces():
            on_prefill(0, stats.prompt_tokens, stats.cached_tokens)
            for processed in self._prefill_steps:
                if not on_prefill(processed, stats.prompt_tokens, stats.cached_tokens):
                    return
            yield from self._script

        return Generation(pieces=pieces(), stats=stats)
