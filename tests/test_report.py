from lmk import bench, report


MACHINE = {"chip": "Apple M3 Ultra", "gpu_cores": 60, "memory_gb": 96, "model_identifier": "Mac15,14", "macos": "26.6.2"}
STATUS = {"build": "v0.7.0", "engine": "7a1e17f3", "config_fingerprint": "abc", "uptime_ms": 5000,
          "model": {"id": "qwen3.8-27b-4bit", "context_length": 262144, "requested_context_length": None, "thinking": False,
                    "reasoning_effort": None, "kv_cache_bits": 8, "speculative_decoding": True},
          "memory": {"pressure": "normal"}, "cache": {"used_bytes": 1}, "draft": {"rounds": 1, "accepted": 2, "drafted": 3},
          "requests": {"answering": 0}, "totals": {"answered": 3}}


def test_the_report_is_one_markdown_block_with_the_scene_and_the_canary_verdict():
    good = bench.Probe(30, 0, 60, 300, 2000, text="1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20")
    text = report.markdown(machine=MACHINE, build="v0.7.0", engine="7a1e17f", config_text="model:\n  name: qwen3.8-27b-4bit\n",
                           config_path="~/.lmk/config.yaml", status=STATUS, canary=good, canary_error=None,
                           events=["10:00:00 INFO LmkReady serving"], stderr=["Traceback (most recent call last):", "ValueError: x"])
    assert text.startswith("<!-- lmk report — paste into https://github.com/seabit-ai/lmk/issues/new")
    assert "- Apple M3 Ultra · 60 GPU cores · 96 GB · Mac15,14 · macOS 26.6.2" in text and "- lmk v0.7.0 · engine 7a1e17f" in text
    assert "## Configuration (`~/.lmk/config.yaml`)" in text and "  name: qwen3.8-27b-4bit" in text
    assert "- matches the reference" in text
    assert "KV cache 8-bit · speculative decoding True" in text
    assert "10:00:00 INFO LmkReady serving" in text and "ValueError: x" in text


def test_the_report_says_when_the_canary_differs_or_could_not_run():
    bad = bench.Probe(30, 0, 60, 300, 2000, text="E anon anonadonaadona")
    text = report.markdown(machine=MACHINE, build="b", engine="e", config_text="", config_path="c", status=STATUS, canary=bad,
                           canary_error=None, events=[], stderr=[])
    assert "**DIFFERS from the reference — this Mac may be producing wrong text**" in text and "E anon anonadonaadona" in text
    down = report.markdown(machine=MACHINE, build="b", engine="e", config_text="", config_path="c", status=None, canary=None,
                           canary_error=None, events=[], stderr=[])
    assert "- not run: lmk is not running" in down and "- lmk is not running (or did not answer)" in down
    failed = report.markdown(machine=MACHINE, build="b", engine="e", config_text="", config_path="c", status=STATUS, canary=None,
                             canary_error="the canary request failed: HTTP 500", events=[], stderr=[])
    assert "- not run: the canary request failed: HTTP 500" in failed
