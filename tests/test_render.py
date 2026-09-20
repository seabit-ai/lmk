import json

from lmk import render

STATUS = {
    "build": "abc1234", "uptime_ms": 11_520_000,
    "model": {"id": "qwen3.8-27b", "path": "/m", "context_length": 262144, "requested_context_length": 262144,
              "input_modalities": ["text", "image"]},
    "cache": {"dir": "/Users/someone/.lmk/cache/0123abcd", "used_bytes": 10 * 1024**3, "max_bytes": 162 * 1024**3,
              "records": 209},
    "in_flight": [],
}


def test_status_says_where_it_is_what_runs_and_how_full_the_cache_is():
    text = render.status_block(STATUS, "http://127.0.0.1:1235")
    assert "✓ lmk is up    http://127.0.0.1:1235/v1   (OpenAI-compatible)" in text
    assert "qwen3.8-27b   (text, image in)" in text
    assert "context    262,144 tokens" in text and "lowered" not in text
    assert "cache      10.0 GB of 162.0 GB   /Users/someone/.lmk/cache" in text
    assert "running    3h 12m   (build abc1234)" in text
    assert "busy       no — idle" in text


def test_a_lowered_context_is_said_out_loud():
    status = json.loads(json.dumps(STATUS))
    status["model"]["context_length"] = 131072
    assert "context    131,072 tokens   (asked for 262,144; lowered to fit this Mac's memory)" in \
        render.status_block(status, "http://x")


def test_a_request_reading_its_prompt_shows_how_far_it_is_and_who_sent_it():
    status = json.loads(json.dumps(STATUS))
    status["in_flight"] = [
        {"purpose": "turn", "ref_id": "s-1/a-4", "phase": "prefill", "running_ms": 12_000,
         "prefill": {"processed": 8192, "total": 27263, "cached": 0}},
        {"purpose": None, "ref_id": None, "phase": "generating", "running_ms": 3_000, "prefill": None},
    ]
    text = render.status_block(status, "http://x")
    assert "busy       2 requests" in text
    assert "reading prompt 8,192 / 27,263 (30%) · turn · s-1/a-4 · 12s" in text
    assert "writing the answer · 3s" in text


def test_connect_block_is_ready_to_paste():
    text = render.connect_block("http://127.0.0.1:1235", "qwen3.8-27b")
    assert "base URL   http://127.0.0.1:1235/v1" in text
    assert "        base_url: http://127.0.0.1:1235" in text          # kitten wants it without /v1
    assert '      local: "lmk:qwen3.8-27b"' in text
    assert "about 3s per 1,000 tokens" in text


def test_not_downloaded_names_the_one_command_to_run():
    assert render.not_downloaded_block("org/m", "not downloaded", 16.1) == \
        "✗ model not downloaded: org/m (16 GB)\n  run:  lmk pull"


def test_a_wildcard_listen_address_is_shown_as_something_a_client_can_dial():
    assert render.base_url("0.0.0.0", 1235) == "http://127.0.0.1:1235"
    assert render.base_url("192.168.1.5", 80) == "http://192.168.1.5:80"


def test_log_lines_lead_with_time_level_event_and_survive_non_json():
    line = render.log_line(json.dumps({"time_ms": 0, "level": "WARN", "event": "LmkContextLowered",
                                       "msg": "shorter", "requested": 5, "inUse": 3}))
    assert line[9:] == "WARN  LmkContextLowered  shorter  requested=5  inUse=3"
    assert render.log_line("plain text\n") == "plain text"


def test_sizes_and_durations():
    assert [render.human_bytes(n) for n in (512, 2048, 5 * 1024**2, 3 * 1024**3, 2 * 1024**4)] == \
        ["512 B", "2 KB", "5.0 MB", "3.0 GB", "2.0 TB"]
    assert [render.human_duration(ms) for ms in (5_000, 65_000, 3_700_000, 90_000_000)] == \
        ["5s", "1m 5s", "1h 1m", "1d 1h"]


def test_a_nearly_full_disk_is_said_next_to_the_cache_line():
    status = json.loads(json.dumps(STATUS))
    status["cache"]["disk_low"] = True
    assert "less than 10 GB free — lmk has stopped adding to the cache" in render.status_block(status, "http://x")
    assert "stopped adding" not in render.status_block(STATUS, "http://x")
