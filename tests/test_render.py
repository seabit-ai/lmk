import json

from lmk import render

STATUS = {
    "build": "abc1234", "uptime_ms": 11_520_000,
    "model": {"id": "qwen3.8-27b-4bit", "path": "/m", "context_length": 262144, "requested_context_length": 262144,
              "input_modalities": ["text", "image"], "thinking": True, "reasoning_effort": None, "kv_cache_bits": 16,
              "speculative_decoding": False},
    "cache": {"dir": "/Users/someone/.lmk/cache/0123abcd", "used_bytes": 10 * 1024**3, "max_bytes": 162 * 1024**3,
              "records": 209},
    "in_flight": [],
}


def test_the_header_says_where_it_is_what_runs_memory_cache_and_limits():
    status = json.loads(json.dumps(STATUS))
    status["memory"] = {"pressure": "normal", "free_percent": 64, "total_bytes": 96 * 1024**3,
                        "lmk_gpu_bytes": int(15.1 * 1024**3), "lmk_gpu_peak_in_use_bytes": 22 * 1024**3}
    status["requests"] = {"answering": 0, "max_parallel": 2, "waiting": 0, "max_queue": 16,
                          "tokens_in_memory": 0, "token_budget": 979_877}
    status["totals"] = {"answered": 412, "refused": 0, "failed": 1, "cancelled": 2,
                        "prompt_tokens": 1_251_870, "cached_tokens": 1_204_113}
    text = render.status_block(status, "http://127.0.0.1:1235")
    assert text.splitlines()[0] == "✓ lmk is up    http://127.0.0.1:1235/v1   (OpenAI-compatible)"
    assert "  model      qwen3.8-27b-4bit · text, image in · 262,144 tokens" in text and "lowered" not in text
    # every switch is stated, off included: a reader must tell "off" from "this lmk has no such thing"
    assert "  settings   thinking on · KV cache 16-bit · speculative decoding off" in text
    off = dict(STATUS, model=dict(STATUS["model"], thinking=False, reasoning_effort="low", kv_cache_bits=8))
    assert "  settings   thinking off (effort low) · KV cache 8-bit · speculative decoding off" in render.status_block(off, "http://127.0.0.1:1235")
    assert "  about it   https://github.com/seabit-ai/lmk/blob/main/docs/models/qwen3.8-27b-4bit.md" in text
    assert "about it" not in render.status_block(dict(STATUS, model=dict(STATUS["model"], id="my-own-model")), "u")
    assert "sampling" not in text                       # an older server without the field: no line
    with_defaults = dict(STATUS, sampling_defaults={"temp": 1.0, "top_p": 0.95, "top_k": 20})
    assert "  sampling   temp 1.0 · top_p 0.95 · top_k 20   (the model's generation_config" in render.status_block(with_defaults, "http://127.0.0.1:1235")
    greedy = dict(STATUS, sampling_defaults={"temp": 0.0})
    assert "  sampling   greedy (temp 0)" in render.status_block(greedy, "http://127.0.0.1:1235")
    assert "  memory     pressure: normal · 64% of 96.0 GB free · lmk holds 15.1 GB" in text
    assert ("  cache      10.0 GB of 162.0 GB in /Users/someone/.lmk/cache · since start 96% of prompt tokens "
            "came from it (1,204,113 of 1,251,870)") in text
    assert "  requests   answering 0 of 2 · waiting 0 of 16 · tokens in memory 0 of 979,877" in text
    assert "  since start  412 answered · 0 refused · 1 failed · 2 cancelled · up 3h 12m · build abc1234" in text
    assert "  idle — no requests" in text and "just finished" not in text


def test_a_lowered_context_is_said_out_loud():
    status = json.loads(json.dumps(STATUS))
    status["model"]["context_length"] = 131072
    assert "131,072 tokens (asked for 262,144; lowered to fit this Mac's memory)" in \
        render.status_block(status, "http://x")


def test_every_request_shows_its_state_and_the_numbers_that_go_with_it():
    status = json.loads(json.dumps(STATUS))
    status["in_flight"] = [
        {"purpose": "turn", "ref_id": "s/step-14", "state": "decode", "part": "thinking", "prompt_tokens": 27190,
         "cached_tokens": 27136, "prefill": None, "completion_tokens": 212, "decode_tokens_per_s": 33.2,
         "running_ms": 7_000},
        {"purpose": "groom", "ref_id": "p/groom", "state": "prefill", "part": None, "prompt_tokens": 14061,
         "cached_tokens": 0, "prefill": {"processed": 8192, "total": 14061, "cached": 0}, "completion_tokens": 0,
         "decode_tokens_per_s": None, "running_ms": 18_000},
        {"purpose": None, "ref_id": None, "state": "starting", "part": None, "prompt_tokens": 900,
         "cached_tokens": None, "prefill": None, "completion_tokens": 0, "decode_tokens_per_s": None,
         "running_ms": 300},
    ]
    status["waiting"] = [{"purpose": "turn", "ref_id": "o/step-2", "waited_ms": 2_000,
                          "reason": "2 requests are being answered (requests.max_parallel)"}]
    lines = render.status_block(status, "http://x").splitlines()
    assert "  decode   thinking   turn · s/step-14  27,190 prompt (27,136 cached) · 212 tokens at 33/s · 7s" in lines
    assert "  prefill  58%        groom · p/groom   8,192 / 14,061 · 0 cached · 18s" in lines
    assert "  starting            (unnamed)         900 prompt · 0s" in lines
    note = " ".join(lines[-2:])                                          # and how to get a name
    assert note.startswith("  (unnamed): the client did not say who it is.")
    assert "X-Lmk-Purpose" in note and "X-Lmk-Ref-Id" in note
    assert "  queued              turn · o/step-2   2s · 2 requests are being answered (requests.max_parallel)" in lines


def test_the_last_answers_stay_on_screen_with_how_they_went():
    status = json.loads(json.dumps(STATUS))
    status["recent"] = [
        {"purpose": "turn", "ref_id": "s/step-13", "outcome": "tool call", "prompt_tokens": 27012,
         "cached_tokens": 26880, "first_token_ms": 1100, "completion_tokens": 349, "decode_tokens_per_s": 33.0,
         "total_ms": 11_700, "ago_ms": 12_000},
        {"purpose": "warmup", "ref_id": None, "outcome": "warmed", "prompt_tokens": 11174, "cached_tokens": 11008,
         "first_token_ms": None, "completion_tokens": 0, "decode_tokens_per_s": None, "total_ms": 900,
         "ago_ms": 3_600_000},
    ]
    lines = render.status_block(status, "http://x").splitlines()
    assert "  just finished" in lines
    assert ("  turn · s/step-13  27,012 prompt (26,880 cached) · first token 1.1s · 349 tokens at 33/s · "
            "tool call · 12s ago") in lines
    assert "  warmup            11,174 prompt (11,008 cached) · warmed · 1h 0m ago" in lines
    assert not any("unnamed" in l for l in lines)   # everyone here said who they are: no note


def test_a_status_from_an_older_lmk_still_renders():
    text = render.status_block({"model": STATUS["model"], "in_flight": [], "uptime_ms": 5000}, "http://x")
    assert "✓ lmk is up" in text and "idle — no requests" in text


def test_connect_block_is_ready_to_paste():
    text = render.connect_block("http://127.0.0.1:1235", {"id": "qwen3.8-27b-4bit", "context_length": 262144,
                                                          "input_modalities": ["text", "image"]})
    assert "base URL   http://127.0.0.1:1235/v1" in text
    assert 'baseUrl: "http://127.0.0.1:1235/v1",' in text              # OpenClaw wants it with /v1
    assert 'id: "qwen3.8-27b-4bit",' in text and "pick the model lmk/qwen3.8-27b-4bit:" in text
    assert 'input: ["text", "image"],' in text and "contextWindow: 262144," in text
    assert "        base_url: http://127.0.0.1:1235" in text          # kitten wants it without /v1
    assert '      local: "lmk:qwen3.8-27b-4bit"' in text
    assert "about 3s per 1,000 tokens" in text


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


def test_memory_is_numbers_only():
    status = json.loads(json.dumps(STATUS))
    status["memory"] = {"pressure": "warning", "free_percent": 12, "total_bytes": 96 * 1024**3,
                        "lmk_gpu_bytes": int(19.2 * 1024**3)}
    text = render.status_block(status, "http://x")
    assert "memory     pressure: warning · 12% of 96.0 GB free · lmk holds 19.2 GB" in text
    assert "another program" not in text  # lmk does not know who uses the memory, so it never says


def test_a_nearly_full_disk_is_said_next_to_the_cache_line():
    status = json.loads(json.dumps(STATUS))
    status["cache"]["disk_low"] = True
    assert "less than 10 GB free — lmk has stopped adding to the cache" in render.status_block(status, "http://x")
    assert "stopped adding" not in render.status_block(STATUS, "http://x")


def test_speculative_decoding_shows_on_the_model_line_and_its_acceptance_below():
    on = dict(STATUS, model=dict(STATUS["model"], speculative_decoding=True),
              draft={"rounds": 100, "accepted": 180, "drafted": 240})
    text = render.status_block(on, "http://127.0.0.1:1235")
    assert "· KV cache 16-bit · speculative decoding on" in text
    assert "  draft      75% of drafted tokens accepted (180 of 240) · 2.8 tokens per round" in text
    fresh = dict(on, draft={"rounds": 0, "accepted": 0, "drafted": 0})
    assert "  draft      no tokens drafted yet" in render.status_block(fresh, "http://127.0.0.1:1235")
    assert "  draft" not in render.status_block(dict(on, draft=None), "http://127.0.0.1:1235")   # no draft model at all
