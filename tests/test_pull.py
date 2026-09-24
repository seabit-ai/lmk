from pathlib import Path

from lmk import pull


class FakeClock:
    def __init__(self):
        self.ms = 0
        self.slept = []

    def mono_ms(self):
        return self.ms

    def wall_ms(self):
        return self.ms

    def sleep_s(self, s):
        self.slept.append(s)
        self.ms += int(s * 1000)


def test_the_line_says_percent_bytes_rate_and_time_left_only_when_measured():
    assert pull.describe(6_800_000_000, 16_100_000_000, 38e6, 245) == "42% · 6.8 of 16.1 GB · 38 MB/s · about 4 min left"
    assert pull.describe(0, 16_100_000_000, None, None) == "0% · 0.0 of 16.1 GB"
    assert pull.describe(16_100_000_000, 16_100_000_000, 40e6, 0) == "100% · 16.1 of 16.1 GB"   # done: no rate, no time left
    assert pull.human_eta(30) == "a minute" and pull.human_eta(2400) == "40 min" and pull.human_eta(5400) == "1.5 h"


def test_the_rate_is_measured_over_the_last_ten_seconds_and_needs_a_second_of_data():
    clock = FakeClock()
    p = pull.Progress(total=1_000_000, clock=clock)
    p.sample(0)
    assert p.rate_bps() is None and p.eta_s(0) is None          # one sample: nothing to say yet
    clock.ms = 500; p.sample(25_000)
    assert p.rate_bps() is None                                  # half a second: too little
    clock.ms = 2_000; p.sample(100_000)
    assert p.rate_bps() == 50_000.0 and p.eta_s(100_000) == 18.0
    for t in range(3_000, 16_000, 1_000):                       # a slower stretch; the window drops the fast start
        clock.ms = t; p.sample(100_000 + (t - 2_000) * 10)
    assert p.rate_bps() == 10_000.0


def test_downloaded_bytes_counts_partial_blobs_so_a_resumed_download_starts_where_it_stopped(tmp_path):
    repo = tmp_path / "models--org--m"
    assert pull.downloaded_bytes(repo) == 0
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "abc").write_bytes(b"x" * 100)
    (repo / "blobs" / "def.incomplete").write_bytes(b"y" * 50)
    assert pull.downloaded_bytes(repo) == 150


def test_download_prints_progress_while_the_worker_fills_the_cache_and_returns_its_path(tmp_path, monkeypatch):
    repo = tmp_path / "models--org--m"
    (repo / "blobs").mkdir(parents=True)
    monkeypatch.setattr(pull, "repo_cache_dir", lambda r: repo)
    clock = FakeClock()
    lines = []

    def fake_snapshot(r):
        for i in range(4):          # the worker writes; the main thread polls between its sleeps
            (repo / "blobs" / f"b{i}").write_bytes(b"x" * 250)
        return str(repo / "snapshots" / "abc")

    path = pull.download("org/m", label="the model", total=1000, say=lines.append, snapshot_download=fake_snapshot,
                         clock=clock, is_tty=False)
    assert path == repo / "snapshots" / "abc"
    assert lines[-1].startswith("  the model · 100% · 0.0 of 0.0 GB")   # the last line says it is complete
    assert all(l.startswith("  the model · ") for l in lines)


def test_a_failed_download_is_reported_not_swallowed(tmp_path, monkeypatch):
    repo = tmp_path / "models--org--m"
    monkeypatch.setattr(pull, "repo_cache_dir", lambda r: repo)

    def boom(r):
        raise OSError("network is down")

    try:
        pull.download("org/m", label="x", total=10, say=lambda _: None, snapshot_download=boom, clock=FakeClock(), is_tty=False)
    except pull.DownloadFailed as e:
        assert "network is down" in str(e)
    else:
        raise AssertionError("expected DownloadFailed")
