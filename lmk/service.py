"""The resident service: one LaunchAgent, installed by `lmk up`, removed by `lmk down`
(design OOBE §E). There is no "stopped but still registered" state."""
import os
import plistlib
import subprocess
from pathlib import Path
from typing import Optional

LABEL = "ai.kitten.lmk"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _domain() -> str:
    return f"gui/{os.getuid()}"


def build_plist(app_dir: Path, log_dir: Path, env: dict) -> dict:
    environment = {"PYTHONPATH": f"{app_dir}/.engine/mlx-engine:{app_dir}"}
    # where config and models were found when `lmk up` ran is where the service must look too
    for key in ("LMK_HOME", "HF_HOME", "HF_HUB_CACHE"):
        if env.get(key):
            environment[key] = env[key]
    return {
        "Label": LABEL,
        "ProgramArguments": [f"{app_dir}/.venv/bin/python", "-P", "-m", "lmk", "serve"],
        "WorkingDirectory": str(app_dir),
        "EnvironmentVariables": environment,
        "RunAtLoad": True,
        # Restart after a crash, never after a clean exit: `lmk serve` exits 0 when it
        # cannot start for a reason a restart will not fix (bad config, model missing,
        # port taken), so launchd does not spin on it.
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "StandardOutPath": str(log_dir / "lmk.stdout.log"),
        "StandardErrorPath": str(log_dir / "lmk.stderr.log"),
    }


def is_registered() -> bool:
    return subprocess.run(["launchctl", "print", f"{_domain()}/{LABEL}"], capture_output=True).returncode == 0


def start(app_dir: Path, log_dir: Path, env: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        plistlib.dump(build_plist(app_dir, log_dir, env), f)
    subprocess.run(["launchctl", "bootstrap", _domain(), str(path)], check=True, capture_output=True)


def stop(sleep, wait_s: int = 120) -> bool:
    """Returns once the old instance is gone — it flushes its cache on the way out, and
    registering a new one before that fails with "Input/output error"."""
    subprocess.run(["launchctl", "bootout", f"{_domain()}/{LABEL}"], capture_output=True)
    for _ in range(wait_s):
        if not is_registered():
            return True
        sleep(1)
    return False


def remove_plist() -> None:
    plist_path().unlink(missing_ok=True)


def parse_lsof_listener(output: str) -> Optional[str]:
    """`lsof -Fpc` prints p<pid> and c<command> lines."""
    pid = command = None
    for line in output.splitlines():
        if line.startswith("p") and pid is None:
            pid = line[1:]
        elif line.startswith("c") and command is None:
            command = line[1:]
    return f"{command or 'a process'} (pid {pid})" if pid else None


def port_owner(port: int) -> Optional[str]:
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_lsof_listener(out)
