from pathlib import Path

import pytest

from lmk.config import ConfigError, load_config


def write(tmp_path, text):
    p = tmp_path / "lmk.yaml"
    p.write_text(text)
    return p


def test_full_config(tmp_path):
    cfg = load_config(write(tmp_path, """
model:
  id: kitten-27b
  path: ~/models/Qwen3.8-27B-MLX-4bit
  context_length: 200000
listen:
  host: 127.0.0.1
  port: 1235
cache:
  dir: ~/somewhere/cache
log:
  dir: /tmp/lmk-logs
"""))
    assert cfg.model.id == "kitten-27b"
    assert cfg.model.path == Path.home() / "models" / "Qwen3.8-27B-MLX-4bit"
    assert cfg.model.context_length == 200000
    assert (cfg.host, cfg.port) == ("127.0.0.1", 1235)
    assert cfg.cache_dir == Path.home() / "somewhere" / "cache"
    assert cfg.log_dir == Path("/tmp/lmk-logs")


def test_model_id_defaults_to_the_directory_name(tmp_path):
    cfg = load_config(write(tmp_path, "model: {path: /m/Qwen3.8-27B-MLX-4bit, context_length: 8}\nlisten: {port: 1}\n"))
    assert cfg.model.id == "qwen3.8-27b-mlx-4bit"
    assert cfg.host == "127.0.0.1"


@pytest.mark.parametrize("text,missing", [
    ("listen: {port: 1}\n", "model.path"),
    ("model: {path: /m}\nlisten: {port: 1}\n", "model.context_length"),
    # no default port: two servers guessing the same number is a silent conflict
    ("model: {path: /m, context_length: 8}\n", "listen.port"),
])
def test_required_fields_have_no_defaults(tmp_path, text, missing):
    with pytest.raises(ConfigError, match=missing):
        load_config(write(tmp_path, text))


def test_missing_file_points_at_the_readme(tmp_path):
    with pytest.raises(ConfigError, match="README"):
        load_config(tmp_path / "nope.yaml")
