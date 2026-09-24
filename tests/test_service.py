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


RUNNING = ("gui/501/ai.kitten.lmk = {\n\tactive count = 1\n\tpath = /h/Library/LaunchAgents/ai.kitten.lmk.plist\n"
           "\tstate = running\n\n\tprogram = /h/.lmk/app/.venv/bin/python\n\truns = 1\n\tpid = 65747\n"
           "\tlast exit code = (never exited)\n\n\tspawn type = daemon (3)\n")
CRASH_LOOP = ("gui/501/ai.kitten.lmk = {\n\tactive count = 0\n\tstate = spawn scheduled\n\n\tminimum runtime = 30\n"
              "\truns = 87\n\tlast exit code = 1\n")
CLEAN_EXIT = "gui/501/ai.kitten.lmk = {\n\tactive count = 0\n\tstate = not running\n\truns = 1\n\tlast exit code = 0\n"


def test_the_job_state_is_read_from_launchctl():
    assert service.parse_launchctl_print(RUNNING) == service.JobState(pid=65747, runs=1, last_exit_code=None)
    assert service.parse_launchctl_print(CRASH_LOOP) == service.JobState(pid=None, runs=87, last_exit_code=1)
    assert service.parse_launchctl_print(CLEAN_EXIT) == service.JobState(pid=None, runs=1, last_exit_code=0)


def test_a_job_that_launchd_has_restarted_or_that_is_gone_is_told_apart_from_one_still_loading():
    assert service.parse_launchctl_print(RUNNING).loading
    assert service.parse_launchctl_print(CRASH_LOOP).crashed
    assert service.parse_launchctl_print(CLEAN_EXIT).exited and not service.parse_launchctl_print(CLEAN_EXIT).crashed
