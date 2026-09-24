"""`lmk serve` must exit 0 (not crash-loop under launchd) when a restart cannot fix the problem."""
import json

import pytest

from lmk import engine, serve
from lmk.models import ResolvedModel


def test_a_missing_model_runtime_ends_in_a_clean_exit_that_says_to_reinstall(tmp_path, monkeypatch, capsys):
    from lmk import modelfit

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LMK_HOME", str(home))
    (home / "config.yaml").write_text(f"model: {{path: {tmp_path}}}\n")
    monkeypatch.setattr(serve, "resolve_model", lambda source: ResolvedModel(path=tmp_path, revision=None))
    monkeypatch.setattr(modelfit, "why_it_does_not_fit", lambda path: None)
    monkeypatch.setattr(engine, "runtime_import_error", lambda: "No module named 'mlx_engine'")
    assert serve.serve() == serve.EXIT_WILL_NOT_FIX_ITSELF
    events = [json.loads(l) for l in capsys.readouterr().err.splitlines() if l.startswith("{")]
    missing = [e for e in events if e["event"] == "LmkRuntimeMissing"]
    assert missing and "mlx_engine" in missing[0]["msg"] and "install" in missing[0]["msg"]
