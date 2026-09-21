"""Refuse a model whose weights alone do not fit this Mac (design memory-guard §D).

Only the weights are compared. A model that fits but leaves little room has its
context lowered by the engine, and `lmk status` says so. There is no override: a
switch that turns a guard off gets turned off."""
from pathlib import Path
from typing import Callable, Optional

from lmk.render import human_bytes

# The reserve mlx-engine's own context fit keeps free (context_fit.py: "The fixed 3 GiB
# reserve covers the largest measured residual between this formula and actual prefill peaks").
RESERVE_BYTES = 3 * 1024**3


def weights_bytes(model_dir: Path) -> int:
    return sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))


def gpu_working_set_bytes() -> tuple[int, int]:
    """(what macOS lets the GPU use, installed memory). Reads the device; loads no model."""
    import mlx.core as mx

    info = mx.device_info()
    return int(info["max_recommended_working_set_size"]), int(info["memory_size"])


def why_it_does_not_fit(model_dir: Path, device: Callable[[], tuple[int, int]] = gpu_working_set_bytes) -> Optional[str]:
    weights = weights_bytes(model_dir)
    working_set, installed = device()
    if weights + RESERVE_BYTES <= working_set:
        return None
    return ("this model does not fit this Mac.\n"
            f"  it needs about {human_bytes(weights)} for its weights; this Mac can give a model {human_bytes(working_set)}\n"
            f"  (macOS keeps the rest of the {human_bytes(installed)} for everything else, and lmk leaves "
            f"{human_bytes(RESERVE_BYTES)} of headroom).\n"
            "  Pick a smaller model, or a lower-bit version of this one.")
