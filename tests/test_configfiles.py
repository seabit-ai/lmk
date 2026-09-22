import yaml

from lmk.configfiles import example_text, refresh_example, seed_config


def test_seed_writes_only_comments_and_never_overwrites(tmp_path):
    path = tmp_path / "config.yaml"
    assert seed_config(path) is True
    assert yaml.safe_load(path.read_text()) is None  # nothing but comments: no default is frozen on disk
    path.write_text("listen: {port: 9}\n")
    assert seed_config(path) is False
    assert path.read_text() == "listen: {port: 9}\n"


def test_example_lists_every_section_and_the_tested_models(tmp_path):
    text = example_text()
    for needle in ("model:", "listen:", "cache:", "log:", "requests:", "max_parallel: 2", "max_queue: 16",
                   "max_wait_seconds: 600", "max_size: 200G", "port: 1235",
                   "qwen3.8-27b", "lmstudio-community/Qwen3.8-27B-MLX-4bit", "16.1 GB"):
        assert needle in text
    assert yaml.safe_load(text) is None


def test_example_is_rewritten_only_when_its_content_differs(tmp_path):
    cfg = tmp_path / "config.yaml"
    example = tmp_path / "config.yaml.example"
    assert refresh_example(cfg) is True
    assert refresh_example(cfg) is False
    example.write_text("# from an older lmk\n")
    assert refresh_example(cfg) is True
    assert example.read_text() == example_text()


def test_every_uncommented_template_line_is_a_valid_config(tmp_path, monkeypatch):
    """The template is the documentation of the schema: uncommenting it must load."""
    from lmk.config import load_config
    from lmk.configfiles import _TEMPLATE

    monkeypatch.setenv("LMK_HOME", str(tmp_path))
    body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in _TEMPLATE.splitlines()[3:])
    body = "\n".join(l for l in body.splitlines() if not l.strip().startswith(("repo:", "path:")))
    cfg = load_config(_write(tmp_path / "c.yaml", body))
    assert cfg.model.id == "qwen3.8-27b-4bit" and cfg.model.context_length == 131072 and cfg.port == 1235
    assert cfg.requests.max_queue == 16


def _write(path, text):
    path.write_text(text)
    return path
