"""Which model the integration tests load: LMK_ITEST_MODEL (a directory), else the
model lmk itself is configured to serve, found the way lmk finds it."""
import os
from pathlib import Path

import pytest


def model_dir() -> Path:
    if os.environ.get("LMK_ITEST_MODEL"):
        return Path(os.environ["LMK_ITEST_MODEL"])
    from lmk.config import load_config
    from lmk.models import ModelNotDownloaded, resolve_model

    try:
        return resolve_model(load_config().model.source).path
    except (ModelNotDownloaded, FileNotFoundError) as e:
        pytest.skip(f"no model to test with: {e} (run `lmk pull`, or set LMK_ITEST_MODEL)")
