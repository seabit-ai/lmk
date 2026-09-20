"""Engine upgrades must not silently cost the user their prompt cache (design OOBE §E′).

The cache identity leaves the engine commit out on purpose. What makes that safe is
this test, run around every engine upgrade:

    make cache-fixture     # BEFORE changing ENGINE_COMMIT: the current engine writes a cache
    ...change ENGINE_COMMIT, copy its requirements.txt, make clean venv...
    make cache-compat      # AFTER: the new engine must restore that cache and say the same thing

If cache-compat fails, bump CACHE_FORMAT_VERSION in lmk/persistcache.py: old caches are
then ignored (and dropped as space is needed) and users pay one cold start.
"""
import json
import os
from pathlib import Path

import pytest

from lmk.engine import engine_commit
from test_itest_persist import probe

pytestmark = pytest.mark.itest


def test_a_cache_written_by_the_previous_engine_is_restored_by_this_one():
    if not os.environ.get("LMK_COMPAT_DIR"):
        pytest.skip("set LMK_COMPAT_DIR (make cache-fixture / make cache-compat)")
    root = Path(os.environ["LMK_COMPAT_DIR"])
    expected_file = root / "expected.json"
    if not expected_file.exists():
        first = probe(root / "cache")
        assert first["cached_tokens"] == 0
        root.mkdir(parents=True, exist_ok=True)
        expected_file.write_text(json.dumps({"written_by_engine_commit": engine_commit(), **first}))
        return  # fixture recorded; the comparison happens on the next run

    expected = json.loads(expected_file.read_text())
    now = probe(root / "cache")
    assert now["prompt_tokens"] == expected["prompt_tokens"], "the prompt tokenizes differently: not a cache problem"
    assert now["cached_tokens"] == (expected["prompt_tokens"] - 1) // 256 * 256, \
        f"cache written by engine {expected['written_by_engine_commit'][:7]} was not restored by {engine_commit()[:7]}"
    assert now["text"] == expected["text"], "restored, but the model says something else: the records are misread"
