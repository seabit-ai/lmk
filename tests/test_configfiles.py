import yaml

from lmk.config import ModelSource, RequestsConfig, load_config
from lmk.configfiles import _home_for_humans, example_text, refresh_example, seed_config
from lmk.models import TESTED_MODELS


def test_the_default_home_is_written_as_tilde_for_humans(monkeypatch):
    monkeypatch.delenv("LMK_HOME", raising=False)
    assert _home_for_humans() == "~/.lmk"
    monkeypatch.setenv("LMK_HOME", "/tmp/elsewhere")
    assert _home_for_humans() == "/tmp/elsewhere"


def test_seed_is_a_real_config_with_every_value_written_out_and_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv("LMK_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    assert seed_config(path) is True
    doc = yaml.safe_load(path.read_text())
    assert set(doc) == {"model", "listen", "cache", "requests", "log"}          # nothing commented away
    assert doc["model"] == {"name": "qwen3.8-27b-4bit", "kv_cache_bits": "auto"} and doc["listen"] == {"host": "127.0.0.1", "port": 1235}
    assert doc["cache"] == {"dir": f"{tmp_path}/cache", "max_size": "200G"} and doc["log"] == {"dir": f"{tmp_path}/logs"}
    assert doc["requests"] == {"max_parallel": 2, "max_queue": 16, "max_wait_seconds": 600}
    assert load_config(path).model.kv_cache_bits_auto        # chosen by the Mac's memory, not pinned for every new user
    path.write_text("listen: {port: 9}\n")
    assert seed_config(path) is False
    assert path.read_text() == "listen: {port: 9}\n"


def test_example_is_a_real_config_with_the_defaults_and_the_default_models_recommendation(tmp_path, monkeypatch):
    """The example loads as it is (so a misspelt key would fail here, not on the user's Mac); its values are
    the defaults, except the two the 27B page recommends (owner, 2026-09-24: "default + 27B")."""
    monkeypatch.setenv("LMK_HOME", str(tmp_path))
    example = tmp_path / "config.yaml.example"
    example.write_text(example_text())
    cfg = load_config(example)
    assert cfg.model.source == ModelSource("name", "qwen3.8-27b-4bit")
    assert (cfg.model.reasoning_effort, cfg.model.kv_cache_bits_auto, cfg.model.speculative_decoding) == ("low", True, True)
    assert (cfg.model.thinking, cfg.model.context_length, cfg.model.draft_tokens) == (None, None, None)
    assert (cfg.host, cfg.port) == ("127.0.0.1", 1235)
    assert (cfg.cache_dir, cfg.cache_max_bytes, cfg.log_dir) == (tmp_path / "cache", 200 * 1024**3, tmp_path / "logs")
    assert cfg.requests == RequestsConfig(2, 16, 600)


def test_example_shows_every_tested_model_and_the_other_ways_of_naming_one_as_commented_lines():
    text = example_text()
    for name in TESTED_MODELS:
        assert f"  # name: {name}\n" in text, name          # the line alone: sizes and speeds live on the model page
    for needle in ("# kv_cache_bits: 16 ", "# kv_cache_bits: 8 ", "# repo: ", "# path: ~/.lmstudio/models/", "# thinking: false", "# reasoning_effort: ", "# port: ",
                   "# max_parallel: "):
        assert needle in text, needle


def test_example_is_rewritten_only_when_its_content_differs(tmp_path):
    cfg = tmp_path / "config.yaml"
    example = tmp_path / "config.yaml.example"
    assert refresh_example(cfg) is True
    assert refresh_example(cfg) is False
    example.write_text("# from an older lmk\n")
    assert refresh_example(cfg) is True
    assert example.read_text() == example_text()
