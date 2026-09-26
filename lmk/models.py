"""The tested-model list, and finding a model on disk.

lmk owns no model directory (design OOBE §B): HuggingFace models live in the
shared HF cache, where every other tool can see them too. Downloading is its
own explicit command (`lmk pull`), never a side effect of starting.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MemoryFit:
    """The engine's context-fit coefficients for this model, read off its load log on the M3 Ultra
    ("Model context auto-fit: ... baseline=... full_kv=... prompt_inputs=... attention_coefficient=...
    rotating_constant=... rotating_per_step=..."). With them, mlx-engine's own formula
    (batched_vision/context_fit.py, `_context_fit_for_step`) says what context fits on any Mac."""
    baseline_gib: float            # GPU memory right after loading, before any conversation
    full_kv_bytes_per_token: int
    prompt_input_bytes_per_token: int
    attention_bytes_per_context_per_step: int
    rotating_constant_gib: float = 0.0
    rotating_bytes_per_prefill_step_token: int = 0
    full_kv_bytes_per_token_8bit: int = 0   # the probe's number with kv_cache_bits: 8; 0 = not measured
    measured_context_on_96gb: int = 0   # what the engine actually fitted on the M3 Ultra 96 GB, when it was less than the maximum


@dataclass(frozen=True)
class Speed:
    """Measured with `lmk bench` on the M3 Ultra 96 GB, one request at a time (docs/benchmarks.md)."""
    prefill_tok_s: int         # a prompt lmk has never seen
    cached_prefill_tok_s: int  # the cached part coming back from disk
    decode_tok_s: float        # writing


@dataclass(frozen=True)
class TestedModel:
    __test__ = False  # not a pytest class

    repo: str
    size_gb: float       # the download
    max_context: int
    images: bool
    thinking: str        # one line: what the switch does on this model
    good_for: str        # one line: why someone would pick it
    fit: MemoryFit
    speed: Speed
    drafts: dict = field(default_factory=dict)  # kind -> repo of a draft model for speculative decoding (mtp / dflash2)
    default_draft: Optional[str] = None         # the model page's pick among `drafts`; model.draft in the config overrides
    kv8_tested: bool = False          # passed the listing gate with kv_cache_bits: 8, so the automatic choice may pick 8

    @property
    def loaded_gib(self) -> float:
        return self.fit.baseline_gib


# Qwen3.8's own MTP draft head, split out of the original weights (the MLX conversions drop it);
# one drafter serves every quantization of the 27B, compatibility is by hidden size and vocabulary.
DRAFT_QWEN38_27B = "seabit-ai/Qwen3.8-27B-MTP-draft"
# z-lab's DFlash 2 drafter for the same model (Inco AI publishes the bf16 original, 4 GB), quantized by
# us to 4 bits (1.1 GB): same acceptance, same output, same speed on the M3 Ultra (research exp07, exp14).
# A block-diffusion drafter that guesses more tokens per round than the MTP head.
DRAFT_QWEN38_27B_DFLASH2 = "seabit-ai/Qwen3.8-27B-DFlash2-4bit"
DRAFTS_QWEN38_27B = {"mtp": DRAFT_QWEN38_27B, "dflash2": DRAFT_QWEN38_27B_DFLASH2}
DRAFT_KINDS = ("mtp", "dflash2")

# Models we have run end to end with a real agent. config.yaml.example and README.md print this
# list, so the one-liners are written for someone choosing a model, not for us.
TESTED_MODELS: dict[str, TestedModel] = {
    "qwen3.8-27b-4bit": TestedModel(
        repo="lmstudio-community/Qwen3.8-27B-MLX-4bit", size_gb=16.1, max_context=262_144, images=True,
        thinking="on by default at the top level; set `reasoning_effort: low` — it tested best",
        good_for="the default; best-tested with `reasoning_effort: low`",
        fit=MemoryFit(14.95, 65536, 10240, 48, full_kv_bytes_per_token_8bit=34816), speed=Speed(323, 53_000, 39.5),
        drafts=DRAFTS_QWEN38_27B, default_draft="dflash2", kv8_tested=True),
    "qwen3.8-27b-8bit": TestedModel(
        repo="lmstudio-community/Qwen3.8-27B-MLX-8bit", size_gb=29.5, max_context=262_144, images=True,
        thinking="same as the 4-bit",
        good_for="the 27B with less quantization loss, 40% slower decode",
        fit=MemoryFit(27.48, 65536, 10240, 48), speed=Speed(319, 44_000, 23.1), drafts=DRAFTS_QWEN38_27B, default_draft="mtp"),
    "qwen3.8-27b-5bit": TestedModel(
        repo="lmstudio-community/Qwen3.8-27B-MLX-5bit", size_gb=19.4, max_context=262_144, images=True,
        thinking="same as the 4-bit", good_for="the 27B between 4- and 8-bit: 19 GB, 20% slower decode than 4-bit",
        fit=MemoryFit(18.08, 65536, 10240, 48), speed=Speed(315, 57_000, 31.6), drafts=DRAFTS_QWEN38_27B, default_draft="mtp"),
    "qwen3.8-27b-6bit": TestedModel(
        repo="lmstudio-community/Qwen3.8-27B-MLX-6bit", size_gb=22.8, max_context=262_144, images=True,
        thinking="same as the 4-bit", good_for="the 27B at 6-bit: 23 GB, 30% slower decode than 4-bit",
        fit=MemoryFit(21.22, 65536, 10240, 48), speed=Speed(315, 53_000, 28.1), drafts=DRAFTS_QWEN38_27B, default_draft="mtp"),
    "qwen3.5-122b-a10b-4bit": TestedModel(
        repo="mlx-community/Qwen3.5-122B-A10B-4bit", size_gb=69.6, max_context=262_144, images=True,
        thinking="on/off only; **use `thinking: false`** — on, it can think for thousands of tokens on a small task",
        good_for="the biggest here; MoE, faster than the 27B but not smarter on our tests",
        fit=MemoryFit(64.82, 24576, 6144, 64, measured_context_on_96gb=165_888), speed=Speed(753, 89_000, 60.5)),
    "qwen3.5-122b-a10b-48gb": TestedModel(
        repo="baa-ai/Qwen3.5-122B-A10B-RAM-48GB-MLX", size_gb=47.2, max_context=262_144, images=True,
        thinking="same as the 4-bit: use `thinking: false`",
        good_for="the 122B for 64 GB Macs (2–3 bit experts, community quant)",
        fit=MemoryFit(43.92, 24576, 6144, 64), speed=Speed(746, 89_000, 53.7)),
    "gemma-4-26b-a4b-4bit": TestedModel(
        repo="mlx-community/gemma-4-26b-a4b-it-4bit", size_gb=15.6, max_context=262_144, images=True,
        thinking="off by default; `thinking: true` lets the model decide per turn, and even off it sometimes thinks briefly",
        good_for="fastest here by far; MoE, 4B active",
        fit=MemoryFit(14.29, 20480, 5632, 32, 0.20, 204800), speed=Speed(1_833, 83_000, 119.5)),
    "gemma-4-e4b-4bit": TestedModel(
        repo="lmstudio-community/gemma-4-E4B-it-MLX-4bit", size_gb=6.8, max_context=131_072, images=True,
        thinking="off by default; same as the 26B-A4B",
        good_for="the small one; 7 GB",
        fit=MemoryFit(6.33, 16384, 26624, 16, 0.02, 40960), speed=Speed(2_199, 183_000, 93.5)),
    "gemma-4-31b-4bit": TestedModel(
        repo="lmstudio-community/gemma-4-31B-it-MLX-4bit", size_gb=18.4, max_context=262_144, images=True,
        thinking="off by default; same as the 26B-A4B",
        good_for="Gemma at the 27B's size; slower, shorter ctx",
        fit=MemoryFit(17.15, 81920, 10752, 64, 0.78, 819200), speed=Speed(252, 26_000, 33.0)),
}
DEFAULT_MODEL_NAME = "qwen3.8-27b-4bit"


def model_page_url(name: str) -> str:
    """Each tested model has a page: what fits, how fast, the recommended config, what to know."""
    return f"https://github.com/seabit-ai/lmk/blob/main/docs/models/{name}.md"


# Apple gives the GPU about this share of unified memory (macOS's default wired limit): on the
# M3 Ultra 96 GB the engine reports a 77.76 GiB working set = 0.81. Other sizes are assumed alike.
GPU_SHARE_OF_MEMORY = 0.81
GIB = 1024 ** 3
ENGINE_RESERVE_BYTES = 3 * GIB           # context_fit.MIN_RUNTIME_RESERVE_BYTES
ENGINE_MIN_CONTEXT = 4_096               # context_fit.MIN_FITTED_CONTEXT_TOKENS
ENGINE_ALLOCATION_STEP = 256
SMALLEST_PREFILL_STEP = 512              # the engine keeps the largest context its smallest step allows
# A model that loads with room for only a few thousand tokens is no use to an agent.
MIN_USEFUL_CONTEXT = 32_768
MAC_MEMORY_SIZES_GB = (16, 24, 32, 48, 64, 96, 128, 192, 256, 512)

# model.kv_cache_bits when config.yaml does not set it (owner, 2026-09-25): 16-bit from this Mac size up, 8-bit below.
# 16-bit decodes up to 44% faster at 128k (research/2026-09-25-spec-long-context SLC-007); below 96 GB it costs too
# much of the context one request can really use — 64 GB 191k instead of 262k, 32 GB 44k instead of 62k
# (research/2026-09-25-kv-memory KVM-007, extrapolated: the line is measured on one 96 GB Mac only).
KV16_FROM_MAC_GB = 96


def mac_memory_gb(total_bytes: int) -> int:
    """The size Apple sells the Mac as: hw.memsize is a whole number of GiB."""
    return round(total_bytes / GIB)


def auto_kv_cache_bits(mac_gb: int, tested: Optional[TestedModel]) -> int:
    """8 only where it was tested: a model not in the list, or listed but never run at 8 bits, keeps 16."""
    if mac_gb >= KV16_FROM_MAC_GB or tested is None or not tested.kv8_tested:
        return 16
    return 8


def context_on(m: TestedModel, mac_gb: int, kv_cache_bits: Optional[int] = None) -> int:
    """The context mlx-engine would fit on a Mac with this much memory: its own formula with this
    model's measured coefficients (one small term it does not log is left out; on the 122B this
    gives 168k where the engine fitted 165,888). 0 = does not load. Expected, not measured, except
    on the 96 GB Mac everything was measured on."""
    f = m.fit
    if not f.baseline_gib:
        return 0          # coefficients not measured yet: nothing to say
    if mac_gb == 96 and f.measured_context_on_96gb:
        return f.measured_context_on_96gb
    working_set = mac_gb * GPU_SHARE_OF_MEMORY * GIB
    fixed = f.baseline_gib * GIB + f.rotating_constant_gib * GIB + f.rotating_bytes_per_prefill_step_token * SMALLEST_PREFILL_STEP
    available = working_set - ENGINE_RESERVE_BYTES - fixed
    if available <= 0:
        return 0
    bits = auto_kv_cache_bits(mac_gb, m) if kv_cache_bits is None else kv_cache_bits
    kv_bytes = f.full_kv_bytes_per_token
    if bits == 8 and f.full_kv_bytes_per_token_8bit:
        kv_bytes = f.full_kv_bytes_per_token_8bit
    per_token = kv_bytes + f.prompt_input_bytes_per_token + f.attention_bytes_per_context_per_step * SMALLEST_PREFILL_STEP
    tokens = int(available // per_token) // ENGINE_ALLOCATION_STEP * ENGINE_ALLOCATION_STEP
    return min(m.max_context, max(ENGINE_MIN_CONTEXT, tokens))


def smallest_mac_gb(m: TestedModel) -> int:
    """The smallest Mac this model is worth running on: the engine's fit leaves at least
    MIN_USEFUL_CONTEXT tokens. Expected, not measured: we have run everything on one 96 GB Mac."""
    for gb in MAC_MEMORY_SIZES_GB:
        if context_on(m, gb) >= MIN_USEFUL_CONTEXT:
            return gb
    return MAC_MEMORY_SIZES_GB[-1]


def _k(n: int) -> str:
    return f"{n // 1000}k" if n >= 1000 else str(n)


def tested_models_markdown() -> str:
    """The "Models" section of README.md: one table per Mac size, smallest first, written for
    someone choosing a model. A unit test holds the README to it, and another holds every row to a
    page under docs/models/."""
    tiers: dict[int, list[str]] = {}
    for name, m in TESTED_MODELS.items():
        tiers.setdefault(smallest_mac_gb(m), []).append(name)
    out = []
    footnote = [False]
    for gb in sorted(tiers):
        # Mac sizes from this group's minimum up to where every model in it reaches its maximum context
        sizes = [g for g in MAC_MEMORY_SIZES_GB if g >= gb]
        last = max(next((i for i, g in enumerate(sizes) if context_on(TESTED_MODELS[n], g) >= TESTED_MODELS[n].max_context),
                        len(sizes) - 1) for n in tiers[gb])
        sizes = sizes[:last + 1]
        out += [f"### Needs at least {gb} GB", "",
                f"| model | good for | ctx size on {' / '.join(f'{g} GB' for g in sizes)} | images | tok/s: cache hit / miss / decode |",
                "|---|---|---|---|---|"]
        for name in tiers[gb]:
            m = TESTED_MODELS[name]
            default = " (default)" if name == DEFAULT_MODEL_NAME else ""
            cells = []
            for g in sizes:
                cell = _k(context_on(m, g))
                if auto_kv_cache_bits(g, m) != 16 and context_on(m, g, 16) != context_on(m, g):
                    cell += f" ({_k(context_on(m, g, 16))} at 16-bit)"
                    footnote[0] = True
                cells.append(cell)
            ctx = " / ".join(cells) + " tokens"
            out.append(f"| [`{name}`](docs/models/{name}.md){default} | {m.good_for} | {ctx} | "
                       f"{'yes' if m.images else 'no'} | "
                       f"{_k(m.speed.cached_prefill_tok_s)} / {m.speed.prefill_tok_s:,} / {m.speed.decode_tok_s:.0f} |")
        out.append("")
    if footnote[0]:
        out.append(f"Below {KV16_FROM_MAC_GB} GB lmk keeps the KV cache at 8 bits for a model tested that way (automatic, "
                   "`kv_cache_bits` in the config overrides it); where that changes the number, the figure at 16 bits is in "
                   "parentheses. Its page says what the setting costs.")
    return "\n".join(out).rstrip("\n")


class DraftNotDownloaded(Exception):
    def __init__(self, repo: str):
        super().__init__(f"the draft model {repo} is not downloaded")
        self.repo = repo


def draft_kind_for(source, kind: Optional[str] = None) -> Optional[str]:
    """The drafter kind in effect: the configured one, else the model page's default; None for
    repo / path sources and for tested models without a draft."""
    if source.kind != "name":
        return None
    tested = TESTED_MODELS[source.value]
    chosen = kind or tested.default_draft
    return chosen if chosen in tested.drafts else None


def draft_repo_for(source, kind: Optional[str] = None) -> Optional[str]:
    """The draft model lmk knows for this source and kind (model.draft; the model page's default
    when unset); None for repo / path sources and for tested models without one."""
    effective = draft_kind_for(source, kind)
    return None if effective is None else TESTED_MODELS[source.value].drafts[effective]


def resolve_draft(source, kind: Optional[str] = None) -> Optional[Path]:
    """The downloaded draft model's directory; None when this model has none. Never
    touches the network."""
    repo = draft_repo_for(source, kind)
    if repo is None:
        return None
    snapshot = _hf_snapshot_dir(repo)
    if snapshot is None or not (snapshot / "config.json").exists() or not any(snapshot.glob("*.safetensors")):
        raise DraftNotDownloaded(repo)
    return snapshot


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
