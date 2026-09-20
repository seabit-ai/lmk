"""K4 acceptance: the prefix cache survives the death of the process."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.itest

from itest_model import model_dir


def probe(cache_dir):
    out = subprocess.run([sys.executable, str(Path(__file__).parent / "persist_probe.py"), str(model_dir()), str(cache_dir)],
                         capture_output=True, text=True, timeout=900, env=os.environ.copy())
    line = next((l for l in out.stdout.splitlines() if l.startswith("PROBE ")), None)
    assert line, f"probe produced no result\nstdout: {out.stdout[-800:]}\nstderr: {out.stderr[-1500:]}"
    return json.loads(line[len("PROBE "):])


def test_prefix_cache_survives_a_process_restart(tmp_path):
    first = probe(tmp_path / "cache")
    assert first["cached_tokens"] == 0, "a fresh cache directory starts cold"

    second = probe(tmp_path / "cache")  # a brand-new process: nothing in memory carries over
    assert second["prompt_tokens"] == first["prompt_tokens"]
    # an identical prompt restores up to its last 256-token block (research LMK-002)
    assert second["cached_tokens"] == (first["prompt_tokens"] - 1) // 256 * 256

    records = list((tmp_path / "cache").rglob("*.safetensors"))
    assert records and (records[0].parent / "layout.json").exists()
