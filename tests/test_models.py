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
