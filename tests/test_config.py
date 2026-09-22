from pathlib import Path

import pytest

from lmk.config import ConfigError, load_config, parse_size


def write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_no_file_at_all_gives_a_config_that_can_run(tmp_path, monkeypatch):
    monkeypatch.setenv("LMK_HOME", str(tmp_path / "home"))
    cfg = load_config()
    assert cfg.model.source.kind == "name"
    assert cfg.model.source.repo == "lmstudio-community/Qwen3.8-27B-MLX-4bit"
    assert cfg.model.id == "qwen3.8-27b-4bit"
    assert cfg.model.context_length is None  # the model's own maximum
    assert (cfg.host, cfg.port) == ("127.0.0.1", 1235)
    assert cfg.cache_dir == tmp_path / "home" / "cache"
    assert cfg.cache_max_bytes == 200 * 1024**3
    assert cfg.log_dir == tmp_path / "home" / "logs"
    assert (cfg.requests.max_parallel, cfg.requests.max_queue, cfg.requests.max_wait_seconds) == (2, 16, 600)


def test_the_seeded_all_comment_file_behaves_like_no_file(tmp_path, monkeypatch):
    from lmk.configfiles import seed_config

    monkeypatch.setenv("LMK_HOME", str(tmp_path))
    seed_config(tmp_path / "config.yaml")
    assert load_config() == load_config(tmp_path / "missing.yaml")


def test_full_config(tmp_path):
    cfg = load_config(write(tmp_path, """
model:
  path: ~/models/Qwen3.8-27B-MLX-4bit
  context_length: 200000
listen:
  host: 0.0.0.0
  port: 4000
cache:
  dir: ~/somewhere/cache
  max_size: 50G
log:
  dir: /tmp/lmk-logs
"""))
    assert cfg.model.id == "qwen3.8-27b-mlx-4bit"      # the directory name: there is no separate id
    assert cfg.model.source.kind == "path"
    assert cfg.model.source.path == Path.home() / "models" / "Qwen3.8-27B-MLX-4bit"
    assert cfg.model.context_length == 200000
    assert (cfg.host, cfg.port) == ("0.0.0.0", 4000)
    assert cfg.cache_dir == Path.home() / "somewhere" / "cache"
    assert cfg.cache_max_bytes == 50 * 1024**3
    assert cfg.log_dir == Path("/tmp/lmk-logs")


def test_any_hf_repo_is_accepted_and_its_id_is_the_repo_name_lowercased(tmp_path):
    cfg = load_config(write(tmp_path, "model: {repo: some-org/Some-Model-MLX-4bit}"))
    assert cfg.model.source.repo == "some-org/Some-Model-MLX-4bit"
    assert cfg.model.id == "some-model-mlx-4bit"


def test_naming_the_model_two_ways_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="keep only one of them"):
        load_config(write(tmp_path, "model: {name: qwen3.8-27b-4bit, path: /x}"))


def test_an_unknown_name_points_at_the_list_and_at_repo(tmp_path):
    with pytest.raises(ConfigError, match=r"not in the tested list \(qwen3.8-27b-4bit, qwen3.8-27b-8bit, qwen3.5-122b-a10b-4bit, qwen3.5-122b-a10b-48gb\).*model.repo"):
        load_config(write(tmp_path, "model: {name: llama-9000}"))


def test_a_scalar_where_a_section_belongs_is_explained(tmp_path):
    with pytest.raises(ConfigError, match="model must be a section"):
        load_config(write(tmp_path, "model: qwen3.8-27b-4bit"))


def test_context_length_must_be_positive(tmp_path):
    with pytest.raises(ConfigError, match="context_length must be positive"):
        load_config(write(tmp_path, "model: {context_length: 0}"))


@pytest.mark.parametrize("text,want", [("200G", 200 * 1024**3), ("200GB", 200 * 1024**3), ("1.5T", int(1.5 * 1024**4)),
                                       ("500m", 500 * 1024**2), ("50GiB", 50 * 1024**3), (1024, 1024)])
def test_sizes(text, want):
    assert parse_size(text) == want


def test_an_unreadable_size_says_how_to_write_one():
    with pytest.raises(ConfigError, match="like 200G"):
        parse_size("lots")


def test_requests_section(tmp_path):
    cfg = load_config(write(tmp_path, "requests: {max_parallel: 4, max_queue: 3, max_wait_seconds: 30}"))
    assert (cfg.requests.max_parallel, cfg.requests.max_queue, cfg.requests.max_wait_seconds) == (4, 3, 30)


@pytest.mark.parametrize("bad", ["0", "-1", "1.5", "two", "true"])
def test_request_limits_must_be_whole_numbers_of_one_or_more(tmp_path, bad):
    with pytest.raises(ConfigError, match=r"requests.max_parallel must be a whole number, 1 or more"):
        load_config(write(tmp_path, f"requests: {{max_parallel: {bad}}}"))


def test_a_leftover_model_id_is_refused_and_told_what_the_name_is_now(tmp_path):
    with pytest.raises(ConfigError, match=r"model.id is gone.*'qwen3.8-27b-8bit'.*Remove the id: line"):
        load_config(write(tmp_path, "model: {name: qwen3.8-27b-8bit, id: whatever}"))


def test_thinking_and_effort_are_server_constants_handed_to_the_template(tmp_path):
    cfg = load_config(write(tmp_path, "model: {name: qwen3.8-27b-4bit}"))
    assert cfg.model.template_kwargs() == {}                       # the template's own defaults
    cfg = load_config(write(tmp_path, "model: {name: qwen3.8-27b-4bit, thinking: false, reasoning_effort: low}"))
    assert cfg.model.template_kwargs() == {"enable_thinking": False, "reasoning_effort": "low"}
    with pytest.raises(ConfigError, match="model.thinking must be true or false"):
        load_config(write(tmp_path, "model: {name: qwen3.8-27b-4bit, thinking: sometimes}"))
