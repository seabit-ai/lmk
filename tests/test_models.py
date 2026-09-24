import json

import pytest

from lmk.config import ModelSource
from lmk.models import missing_weight_files, native_context_length, resolve_model


def test_a_local_directory_resolves_to_itself_without_a_revision(tmp_path):
    resolved = resolve_model(ModelSource("path", str(tmp_path)))
    assert resolved.path == tmp_path and resolved.revision is None


def test_a_missing_local_directory_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.path does not exist"):
        resolve_model(ModelSource("path", str(tmp_path / "nope")))


def test_missing_shards_are_found_from_the_weight_index(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert missing_weight_files(tmp_path) == ["model-00002-of-00002.safetensors"]
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"x")
    assert missing_weight_files(tmp_path) == []


def test_a_single_file_model_needs_no_index(tmp_path):
    assert missing_weight_files(tmp_path) == ["*.safetensors"]
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert missing_weight_files(tmp_path) == []


def test_native_context_length_prefers_text_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"max_position_embeddings": 4096, "text_config": {"max_position_embeddings": 262144}}))
    assert native_context_length(tmp_path) == 262144
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
    assert native_context_length(tmp_path) == 32768
    assert native_context_length(tmp_path / "nope") is None


def hf_cache(tmp_path, monkeypatch, repo="org/Some-Model", commit="abc123", shards=("model.safetensors",)):
    import huggingface_hub.constants as constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    repo_dir = tmp_path / "hub" / ("models--" + repo.replace("/", "--"))
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(commit)
    if shards is not None:
        snapshot = repo_dir / "snapshots" / commit
        snapshot.mkdir(parents=True)
        for shard in shards:
            (snapshot / shard).write_bytes(b"x")
    return repo_dir


def test_a_downloaded_hf_model_resolves_to_its_snapshot_and_commit(tmp_path, monkeypatch):
    repo_dir = hf_cache(tmp_path, monkeypatch)
    resolved = resolve_model(ModelSource("repo", "org/Some-Model"))
    assert resolved.path == repo_dir / "snapshots" / "abc123" and resolved.revision == "abc123"


def test_a_repo_that_was_only_looked_at_is_not_downloaded(tmp_path, monkeypatch):
    """What LM Studio leaves in the HF cache: refs/main, no snapshot."""
    from lmk.models import ModelNotDownloaded

    hf_cache(tmp_path, monkeypatch, shards=None)
    with pytest.raises(ModelNotDownloaded, match="not downloaded"):
        resolve_model(ModelSource("repo", "org/Some-Model"))
    with pytest.raises(ModelNotDownloaded, match="not downloaded"):
        resolve_model(ModelSource("repo", "never/heard-of-it"))


def test_an_interrupted_download_is_reported_as_incomplete(tmp_path, monkeypatch):
    from lmk.models import ModelNotDownloaded

    repo_dir = hf_cache(tmp_path, monkeypatch, shards=("model-00001-of-00002.safetensors",))
    (repo_dir / "snapshots" / "abc123" / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    with pytest.raises(ModelNotDownloaded, match=r"download incomplete \(1 weight file"):
        resolve_model(ModelSource("repo", "org/Some-Model"))


def test_finding_a_model_does_not_go_through_the_function_mlx_engine_disables(tmp_path, monkeypatch):
    import huggingface_hub

    def sabotaged(*args, **kwargs):
        raise RuntimeError("Cannot proceed without downloading from huggingface")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", sabotaged)
    hf_cache(tmp_path, monkeypatch)
    assert resolve_model(ModelSource("repo", "org/Some-Model")).revision == "abc123"


def test_the_readme_models_table_is_the_tested_list():
    from pathlib import Path

    from lmk.models import tested_models_markdown

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert tested_models_markdown() in readme, "README.md 'Models' table is out of date: paste tested_models_markdown()"


def test_every_tested_model_has_its_own_page():
    from pathlib import Path

    from lmk.models import TESTED_MODELS

    for name in TESTED_MODELS:
        page = Path(__file__).resolve().parent.parent / "docs" / "models" / f"{name}.md"
        assert page.exists(), f"docs/models/{name}.md is missing: every tested model gets a page"
        text = page.read_text()
        for heading in ("## Fits", "## Speed", "## Recommended configuration", "## Thinking", "## Known issues", "## Tested", "## Not tested"):
            assert heading in text, f"{page.name} lacks '{heading}'"
        from lmk.models import smallest_mac_gb
        assert f"| needs at least (expected, not tested) | {smallest_mac_gb(TESTED_MODELS[name])} GB" in text, \
            f"{page.name}: the smallest-Mac row disagrees with smallest_mac_gb()"


def test_models_are_grouped_by_the_smallest_mac_with_a_useful_context():
    from lmk.models import TESTED_MODELS, context_on, smallest_mac_gb, tested_models_markdown

    m122 = TESTED_MODELS["qwen3.5-122b-a10b-4bit"]
    assert context_on(m122, 96) == 165_888                  # measured on our 96 GB Mac wins over the formula
    assert context_on(m122, 64) == 0 and smallest_mac_gb(m122) == 96
    m27 = TESTED_MODELS["qwen3.8-27b-4bit"]
    assert context_on(m27, 96) == 262_144 and context_on(m27, 24) < 32_768 and smallest_mac_gb(m27) == 32
    md = tested_models_markdown()
    assert md.index("### Needs at least 32 GB") < md.index("### Needs at least 48 GB") < md.index("### Needs at least 96 GB")
    assert "| ctx size on 32 GB / 48 GB / 64 GB / 96 GB |" in md   # up to where every model in the group maxes out
    assert "| 122k (85k at 16-bit) / 262k (223k at 16-bit) / 262k / 262k tokens |" in md   # the 8-bit recommendation, and what 16-bit gives
    assert md.rstrip().endswith("its page says what the setting costs.")
    assert "| 165k / 262k tokens |" in md                                # 96 GB measured, 128 GB from the formula


def _draft_in_cache(tmp_path, monkeypatch, complete=True):
    import huggingface_hub.constants as constants
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo_dir = tmp_path / "models--seabit-ai--Qwen3.8-27B-MTP-draft"
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text("d1")
    snap = repo_dir / "snapshots" / "d1"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text('{"model_type": "qwen3_5_mtp"}')
    if complete:
        (snap / "model.safetensors").write_bytes(b"w")
    return snap


def test_the_27b_has_a_draft_and_other_sources_have_none(tmp_path, monkeypatch):
    from lmk.models import DraftNotDownloaded, draft_repo_for, resolve_draft
    assert draft_repo_for(ModelSource("name", "qwen3.8-27b-4bit")) == "seabit-ai/Qwen3.8-27B-MTP-draft"
    assert draft_repo_for(ModelSource("name", "gemma-4-e4b-4bit")) is None
    assert draft_repo_for(ModelSource("repo", "org/Some-Model")) is None
    assert resolve_draft(ModelSource("repo", "org/Some-Model")) is None
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(DraftNotDownloaded, match="seabit-ai/Qwen3.8-27B-MTP-draft is not downloaded"):
        resolve_draft(ModelSource("name", "qwen3.8-27b-4bit"))
    _draft_in_cache(tmp_path, monkeypatch, complete=False)
    with pytest.raises(DraftNotDownloaded):
        resolve_draft(ModelSource("name", "qwen3.8-27b-4bit"))
    snap = _draft_in_cache(tmp_path, monkeypatch)
    assert resolve_draft(ModelSource("name", "qwen3.8-27b-4bit")) == snap
