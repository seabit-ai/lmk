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


# design 2026-10-20-draft-for-any-model: a draft the config asks for but cannot get ends in a clean exit that says why
def _repo_model_with_speculative(tmp_path, monkeypatch, extra=""):
    from lmk import modelfit

    home = tmp_path / "home"
    home.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 64, "vocab_size": 100}}))
    monkeypatch.setenv("LMK_HOME", str(home))
    (home / "config.yaml").write_text(f"model: {{repo: org/Some-Model, speculative_decoding: true{extra}}}\n")
    monkeypatch.setattr(serve, "resolve_model", lambda source: ResolvedModel(path=model, revision="r1"))
    monkeypatch.setattr(modelfit, "why_it_does_not_fit", lambda path: None)
    return model


def _events(capsys):
    return [json.loads(l) for l in capsys.readouterr().err.splitlines() if l.startswith("{")]


def test_a_model_without_its_own_head_ends_in_a_clean_exit_that_names_the_ways_out(tmp_path, monkeypatch, capsys):
    _repo_model_with_speculative(tmp_path, monkeypatch)
    assert serve.serve() == serve.EXIT_WILL_NOT_FIX_ITSELF
    unusable = [e for e in _events(capsys) if e["event"] == "LmkDraftUnusable"]
    assert unusable and "no MTP head" in unusable[0]["msg"] and "`official`" in unusable[0]["msg"]


def test_a_draft_made_for_another_model_ends_in_a_clean_exit_with_both_numbers(tmp_path, monkeypatch, capsys):
    _repo_model_with_speculative(tmp_path, monkeypatch, extra=f", draft_weights: {tmp_path}/draft")
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps({"model_type": "qwen3_5_mtp", "text_config": {"hidden_size": 5120, "vocab_size": 100}}))
    assert serve.serve() == serve.EXIT_WILL_NOT_FIX_ITSELF
    unusable = [e for e in _events(capsys) if e["event"] == "LmkDraftUnusable"]
    assert unusable and "5120" in unusable[0]["msg"] and "64" in unusable[0]["msg"]


def test_a_draft_folder_that_is_not_there_is_named(tmp_path, monkeypatch, capsys):
    _repo_model_with_speculative(tmp_path, monkeypatch, extra=f", draft_weights: {tmp_path}/nope")
    assert serve.serve() == serve.EXIT_WILL_NOT_FIX_ITSELF
    unusable = [e for e in _events(capsys) if e["event"] == "LmkDraftUnusable"]
    assert unusable and "nope" in unusable[0]["msg"]
