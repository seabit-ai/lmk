from pathlib import Path

from lmk import service


def test_plist_runs_serve_from_the_app_dir_and_restarts_only_after_a_crash():
    plist = service.build_plist(Path("/h/.lmk/app"), Path("/h/.lmk/logs"), {"LMK_HOME": "/h/.lmk", "PATH": "/bin"})
    assert plist["Label"] == "ai.kitten.lmk"
    assert plist["ProgramArguments"] == ["/h/.lmk/app/.venv/bin/python", "-P", "-m", "lmk", "serve"]
    assert plist["EnvironmentVariables"] == {"PYTHONPATH": "/h/.lmk/app/.engine/mlx-engine:/h/.lmk/app",
                                             "LMK_HOME": "/h/.lmk"}
    assert plist["RunAtLoad"] is True and plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["StandardErrorPath"] == "/h/.lmk/logs/lmk.stderr.log"


def test_where_models_were_found_is_passed_on_to_the_service():
    env = service.build_plist(Path("/a"), Path("/l"), {"HF_HOME": "/big/hf"})["EnvironmentVariables"]
    assert env["HF_HOME"] == "/big/hf" and "LMK_HOME" not in env


def test_the_process_holding_a_port_is_named():
    assert service.parse_lsof_listener("p528\ncPython\nf9\n") == "Python (pid 528)"
    assert service.parse_lsof_listener("") is None
