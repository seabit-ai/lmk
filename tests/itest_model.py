"""Which model the integration tests load: LMK_ITEST_MODEL (a directory), else the
model lmk itself is configured to serve, found the way lmk finds it.
LMK_ITEST_KV_BITS=8|4 runs the same acceptance with a quantized KV cache (`model.kv_cache_bits`).
LMK_ITEST_DRAFT=<dir> runs it with that draft model loaded (`model.speculative_decoding: true`)."""
import os
from pathlib import Path

import pytest


def kv_cache_bits() -> int:
    return int(os.environ.get("LMK_ITEST_KV_BITS") or 16)


def draft_path():
    return Path(os.environ["LMK_ITEST_DRAFT"]) if os.environ.get("LMK_ITEST_DRAFT") else None


def model_dir() -> Path:
    if os.environ.get("LMK_ITEST_MODEL"):
        return Path(os.environ["LMK_ITEST_MODEL"])
    from lmk.config import load_config
    from lmk.models import ModelNotDownloaded, resolve_model

    try:
        return resolve_model(load_config().model.source).path
    except (ModelNotDownloaded, FileNotFoundError) as e:
        pytest.skip(f"no model to test with: {e} (run `lmk pull`, or set LMK_ITEST_MODEL)")
