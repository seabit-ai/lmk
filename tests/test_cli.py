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


# ---- the wait in `lmk up`: what it shows while the model loads, and how fast it gives up ----

class _Clock:
    def __init__(self): self.now = 0; self.slept = 0
    def wall_ms(self): return self.now
    def mono_ms(self): return self.now
    def sleep_s(self, s): self.now += int(s * 1000); self.slept += s


class _Memory:
    def __init__(self, resident): self.resident = resident
    def resident_bytes(self, pid): return self.resident


@pytest.fixture
def starting(home, tmp_path, monkeypatch):
    """A model on disk, nothing listening, and a service that `lmk up` will start and then watch."""
    from lmk import clock, memory, modelfit, render

    home.mkdir(parents=True)
    (home / "config.yaml").write_text(f"model: {{path: {tmp_path}}}\n")
    monkeypatch.setattr(modelfit, "why_it_does_not_fit", lambda path: None)
    monkeypatch.setattr(modelfit, "weights_bytes", lambda path: 16 * 1024**3)
    monkeypatch.setattr(service, "start", lambda *a: None)
    monkeypatch.setattr(cli, "_smoke", lambda cfg: None)
    monkeypatch.setattr(render, "status_block", lambda status, url: "STATUS")
    monkeypatch.setattr(render, "connect_block", lambda url, model: "CONNECT")
    fake = _Clock()
    previous = (clock.get_current_clock(), memory.get_current_memory())
    clock.set_current_clock(fake)
    memory.set_current_memory(_Memory(int(8.2 * 1024**3)))
    yield fake
    clock.set_current_clock(previous[0])
    memory.set_current_memory(previous[1])


def _answers_after(polls: int):
    seen = {"n": 0}

    def status(cfg, timeout=2.0):
        seen["n"] += 1
        return {"model": {"id": "m"}, "in_flight": []} if seen["n"] > polls else None
    return status


def test_up_shows_how_much_of_the_model_is_in_memory_never_the_seconds(starting, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_get_status", _answers_after(3))
    monkeypatch.setattr(service, "job_state", lambda: service.JobState(pid=7, runs=1, last_exit_code=None))
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    assert cli.main(["up"]) == 0
    out = capsys.readouterr().out
    assert "8.2 of 16.0 GB" in out
    assert "s\r" not in out and " 1s" not in out and " 2s" not in out


def test_up_reports_a_crash_loop_at_once_instead_of_waiting_ten_minutes(starting, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_get_status", _answers_after(10**6))
    monkeypatch.setattr(service, "job_state", lambda: service.JobState(pid=None, runs=2, last_exit_code=1))
    assert cli.main(["up"]) == 5
    assert "crashed while starting" in capsys.readouterr().err
    assert starting.slept <= 2


def test_up_reports_a_clean_exit_at_once(starting, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_get_status", _answers_after(10**6))
    monkeypatch.setattr(service, "job_state", lambda: service.JobState(pid=None, runs=1, last_exit_code=0))
    assert cli.main(["up"]) == 5
    assert "exited while starting" in capsys.readouterr().err
    assert starting.slept <= 2


def test_status_while_loading_shows_the_bytes_in_memory(starting, capsys, monkeypatch):
    monkeypatch.setattr(service, "is_registered", lambda: True)
    monkeypatch.setattr(service, "job_state", lambda: service.JobState(pid=7, runs=1, last_exit_code=None))
    assert cli.main(["status"]) == 1
    assert "8.2 of 16.0 GB" in capsys.readouterr().out


def test_status_names_a_crash_loop(starting, capsys, monkeypatch):
    monkeypatch.setattr(service, "is_registered", lambda: True)
    monkeypatch.setattr(service, "job_state", lambda: service.JobState(pid=None, runs=5, last_exit_code=1))
    assert cli.main(["status"]) == 1
    out = capsys.readouterr().out
    assert "crashes while starting" in out and "lmk logs" in out


def test_why_it_did_not_start_shows_only_this_start_not_the_previous_life(home, tmp_path):
    from lmk.config import load_config

    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(f"model: {{path: {tmp_path}}}\n")
    cfg = load_config()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    old = '{"time_ms": 1000, "level": "INFO", "event": "LmkReady", "msg": "serving"}'
    stopping = '{"time_ms": 2000, "level": "INFO", "event": "LmkStopping", "msg": "shutting down"}'
    fresh = '{"time_ms": 5000, "level": "ERROR", "event": "LmkRuntimeMissing", "msg": "not installed"}'
    (cfg.log_dir / "lmk.jsonl").write_text(old + "\n" + stopping + "\n" + fresh + "\n")
    # stderr: the previous life's traceback sits between the old lines and this start's line
    (cfg.log_dir / "lmk.stderr.log").write_text(old + "\nTraceback (old)\n  old frame\n" + fresh + "\n")
    text = cli._why_it_did_not_start(cfg, since_ms=3000)
    assert "LmkRuntimeMissing" in text and "LmkReady" not in text and "Traceback (old)" not in text


def test_why_it_did_not_start_keeps_a_traceback_that_came_before_any_log_line(home, tmp_path):
    from lmk.config import load_config

    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(f"model: {{path: {tmp_path}}}\n")
    cfg = load_config()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    stopping = '{"time_ms": 2000, "level": "INFO", "event": "LmkStopping", "msg": "shutting down"}'
    (cfg.log_dir / "lmk.jsonl").write_text(stopping + "\n")
    (cfg.log_dir / "lmk.stderr.log").write_text(stopping + "\nTraceback (most recent call last):\n  ModuleNotFoundError: x\n")
    text = cli._why_it_did_not_start(cfg, since_ms=3000)
    assert "ModuleNotFoundError: x" in text and "LmkStopping" not in text
