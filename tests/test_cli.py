"""The decisions `lmk` makes before it touches launchd. The launchd side is verified live."""
import pytest

from lmk import cli, service


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LMK_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(service, "is_registered", lambda: False)
    monkeypatch.setattr(service, "port_owner", lambda port: None)
    monkeypatch.setattr(service, "start", lambda *a: pytest.fail("must not start the service"))
    monkeypatch.setattr(cli, "app_dir", lambda: tmp_path / "app")
    monkeypatch.setattr(cli, "_get_status", lambda cfg, timeout=2.0: None)  # never talk to a real lmk on this machine
    return tmp_path / "home"


def test_first_run_writes_the_two_config_files(home, capsys):
    cli.main(["status"])
    assert (home / "config.yaml").exists() and (home / "config.yaml.example").exists()
    assert "lmk is not running" in capsys.readouterr().out


def test_up_refuses_to_start_without_the_model_and_names_the_command(home, capsys, monkeypatch):
    from lmk.models import ModelNotDownloaded

    def not_there(source):
        raise ModelNotDownloaded(source.repo, "not downloaded")

    monkeypatch.setattr(cli, "resolve_model", not_there)
    assert cli.main(["up"]) == 3
    assert capsys.readouterr().err == \
        "✗ model not downloaded: lmstudio-community/Qwen3.8-27B-MLX-4bit (16 GB)\n  run:  lmk pull\n"


def test_up_names_whoever_holds_the_port_and_stops(home, tmp_path, capsys, monkeypatch):
    (home).mkdir(parents=True)
    (home / "config.yaml").write_text(f"model: {{path: {tmp_path}}}\nlisten: {{port: 1}}\n")
    monkeypatch.setattr(service, "port_owner", lambda port: "Python (pid 528)")
    assert cli.main(["up"]) == 4
    assert "port 1 is taken by Python (pid 528)" in capsys.readouterr().err


def test_up_refuses_to_run_the_service_from_a_source_checkout(home, tmp_path, capsys):
    (tmp_path / "app" / ".git").mkdir(parents=True)
    assert cli.main(["up"]) == 2
    assert "make install" in capsys.readouterr().err


def test_a_config_mistake_is_reported_with_the_file_it_is_in(home, capsys):
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("model: qwen3.8-27b-4bit\n")
    assert cli.main(["status"]) == 2
    err = capsys.readouterr().err
    assert "config.yaml: model must be a section" in err


def test_pull_has_nothing_to_do_for_a_local_directory(home, tmp_path, capsys):
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(f"model: {{path: {tmp_path}}}\n")
    assert cli.main(["pull"]) == 0
    assert "nothing to download" in capsys.readouterr().out


def test_down_with_nothing_installed_says_so(home, capsys, monkeypatch):
    monkeypatch.setattr(service, "plist_path", lambda: home / "none.plist")
    assert cli.main(["down"]) == 0
    assert "Nothing to do" in capsys.readouterr().out
