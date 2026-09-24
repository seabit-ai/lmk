"""Downloading a model (and its draft) with one progress line, so a user can tell what is taking
this long. `lmk up` downloads what is missing (owner, 2026-09-24: the pull step should not be
one the user has to know about, as long as the progress is a percentage); `lmk pull` does the
same on request. Files go into the shared HuggingFace cache, where other tools can use them,
and an interrupted download resumes from its partial blobs.

Nothing here imports mlx-engine: once imported it replaces huggingface_hub.snapshot_download
with a function that always raises (CLAUDE.md, pitfalls).
"""
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from lmk.clock import get_current_clock


class DownloadFailed(Exception):
    pass


def repo_cache_dir(repo: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def downloaded_bytes(repo_dir: Path) -> int:
    """Bytes on disk for this repo, partial (.incomplete) blobs included: what a resumed download keeps."""
    blobs = repo_dir / "blobs"
    if not blobs.is_dir():
        return 0
    return sum(p.stat().st_size for p in blobs.iterdir() if p.is_file())


def total_bytes(repo: str) -> int:
    """The repo's size on the Hub (network). Decimal bytes, as the Hub and our models table count them."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    return sum(s.size or 0 for s in info.siblings)


def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def human_eta(seconds: float) -> str:
    if seconds < 90:
        return "a minute"
    if seconds < 3600:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def describe(done: int, total: int, rate_bps: Optional[float], eta_s: Optional[float]) -> str:
    """`42% · 6.8 of 16.1 GB · 38 MB/s · about 4 min left` — the rate and the time are measured, never guessed."""
    pct = min(100, done * 100 // total) if total else 0
    parts = [f"{pct}%", f"{done / 1e9:.1f} of {gb(total)}"]
    if rate_bps and done < total:
        parts.append(f"{rate_bps / 1e6:.0f} MB/s")
    if eta_s is not None and done < total:
        parts.append(f"about {human_eta(eta_s)} left")
    return " · ".join(parts)


class Progress:
    """Rate over the last WINDOW_S seconds of samples, and the time left from it."""
    WINDOW_S = 10

    def __init__(self, total: int, clock=None):
        self.total = total
        self.clock = clock or get_current_clock()
        self.samples: deque = deque()

    def sample(self, done: int) -> None:
        now = self.clock.mono_ms()
        self.samples.append((now, done))
        while len(self.samples) > 2 and now - self.samples[0][0] > self.WINDOW_S * 1000:
            self.samples.popleft()

    def rate_bps(self) -> Optional[float]:
        if len(self.samples) < 2:
            return None
        (t0, b0), (t1, b1) = self.samples[0], self.samples[-1]
        if t1 - t0 < 1000 or b1 <= b0:
            return None
        return (b1 - b0) / ((t1 - t0) / 1000)

    def eta_s(self, done: int) -> Optional[float]:
        rate = self.rate_bps()
        return None if not rate else max(0.0, (self.total - done) / rate)

    def line(self, done: int) -> str:
        return describe(done, self.total, self.rate_bps(), self.eta_s(done))


def _snapshot_download(repo: str) -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()  # one line of ours instead of a bar per file
    return snapshot_download(repo)


def download(repo: str, *, label: str, total: int, say: Callable[[str], None] = print,
             snapshot_download: Callable[[str], str] = _snapshot_download, clock=None,
             interval_s: float = 0.5, is_tty: Optional[bool] = None) -> Path:
    """Downloads `repo` while printing one progress line (rewritten in place on a terminal; one
    line per 10% otherwise). Ctrl-C reaches the caller as KeyboardInterrupt and the partial blobs
    stay on disk for the next run. A failed download raises DownloadFailed."""
    clock = clock or get_current_clock()
    repo_dir = repo_cache_dir(repo)
    result: dict = {}

    def run():
        try:
            result["path"] = Path(snapshot_download(repo))
        except BaseException as e:  # noqa: BLE001 - reported to the caller, whatever it was
            result["error"] = e

    worker = threading.Thread(target=run, daemon=True)   # daemon: Ctrl-C in the main thread ends the process
    worker.start()
    progress = Progress(total, clock)
    tty = sys.stdout.isatty() if is_tty is None else is_tty
    shown_pct = -1
    while True:
        alive = worker.is_alive()
        done = downloaded_bytes(repo_dir) if alive else total
        progress.sample(done)
        line = f"  {label} · {progress.line(done)}"
        if tty:
            sys.stdout.write("\r" + line + " " * 12)
            sys.stdout.flush()
        else:
            pct = min(100, done * 100 // total) if total else 100
            if pct // 10 > shown_pct // 10 or not alive:
                say(line)
                shown_pct = pct
        if not alive:
            break
        clock.sleep_s(interval_s)
    if tty:
        sys.stdout.write("\n")
        sys.stdout.flush()
    if "error" in result:
        raise DownloadFailed(str(result["error"]))
    return result["path"]
