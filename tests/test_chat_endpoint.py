import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from lmk.engine import FakeEngine, GenerationStats, LoadedModel
from lmk.server import LmkServer

MODEL = LoadedModel(id="kitten-27b", path=Path("/m/x"), context_length=200000)


class FakeChatFormat:
    """Stands in for the model's template + mlx-lm's parser. The tool-call text
    below is what the real 27B wrote in research exp02."""
    tool_call_start, tool_call_end = "<tool_call>", "</tool_call>"

    def __init__(self):
        self.rendered = []

    def render(self, messages, tools):
        self.rendered.append((messages, tools))
        return "PROMPT<think>\n"

    def starts_in_reasoning(self, prompt_text):
        return prompt_text.rstrip().endswith("<think>")

    def parse_tool_call(self, block, tools):
        if "BROKEN" in block:
            raise ValueError("No function provided.")
        name = block.split("<function=")[1].split(">")[0]
        return {"name": name, "arguments": {"path": "notes.md", "max_lines": 20}}


TOOL_TURN = ["We need one call.\n", "</think>", "\n\n",
             "<tool_call>\n<function=file_read>\n<parameter=path>\nnotes.md\n</parameter>\n</function>\n</tool_call>"]
TEXT_TURN = ["Both are done.\n", "</think>", "\n\n", "Here's what", " I found."]


def serve(script, stats=None, prefill_steps=None):
    fmt = FakeChatFormat()
    engine = FakeEngine(MODEL, chat_format=fmt, script=script,
                        stats=stats or GenerationStats(prompt_tokens=441, cached_tokens=256, completion_tokens=88),
                        prefill_steps=prefill_steps)
    srv = LmkServer(engine, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, engine, fmt


def post(srv, body, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{srv.port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    return urllib.request.urlopen(req, timeout=10)


def stream_chunks(resp):
    lines = [l[len("data: "):] for l in resp.read().decode().split("\n") if l.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(l) for l in lines[:-1]]


def deltas(chunks, key):
    return "".join(c["choices"][0]["delta"].get(key, "") for c in chunks if c["choices"])


def test_streamed_tool_call_in_openai_shape():
    srv, engine, fmt = serve(TOOL_TURN)
    try:
        chunks = stream_chunks(post(srv, {"model": "kitten-27b", "stream": True, "tools": [{"type": "function"}],
                                          "messages": [{"role": "user", "content": "hi"}]}))
    finally:
        srv.shutdown()
    assert deltas(chunks, "reasoning_content") == "We need one call.\n"
    assert deltas(chunks, "content") == ""
    calls = [c["choices"][0]["delta"]["tool_calls"][0] for c in chunks if c["choices"] and "tool_calls" in c["choices"][0]["delta"]]
    assert len(calls) == 1
    assert calls[0]["index"] == 0 and calls[0]["type"] == "function" and calls[0]["id"].startswith("call_")
    assert calls[0]["function"] == {"name": "file_read", "arguments": '{"path": "notes.md", "max_lines": 20}'}
    finishes = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finishes == ["tool_calls"]
    # tools reach the template untouched; the engine gets the rendered prompt
    assert fmt.rendered[0][1] == [{"type": "function"}]
    assert engine.requests[0]["prompt"] == "PROMPT<think>\n"


def test_usage_carries_cache_hits_in_the_standard_field():
    srv, _, _ = serve(TEXT_TURN)
    try:
        chunks = stream_chunks(post(srv, {"model": "kitten-27b", "stream": True, "messages": []}))
    finally:
        srv.shutdown()
    usage = [c["usage"] for c in chunks if c.get("usage")]
    assert usage == [{"prompt_tokens": 441, "completion_tokens": 88, "total_tokens": 529,
                      "prompt_tokens_details": {"cached_tokens": 256}}]
    assert deltas(chunks, "content") == "Here's what I found."
    assert "think>" not in deltas(chunks, "content")


# Prefill progress rides as chunks with `choices: []` — the same trick the usage
# chunk uses — so a stock OpenAI client skips them instead of choking.
def test_prefill_progress_is_invisible_to_stock_clients():
    srv, _, _ = serve(TEXT_TURN, stats=GenerationStats(prompt_tokens=6000, cached_tokens=2048, completion_tokens=5),
                      prefill_steps=[4096, 5999])
    try:
        chunks = stream_chunks(post(srv, {"model": "kitten-27b", "stream": True, "messages": []}))
    finally:
        srv.shutdown()
    prefill = [c["lmk"]["prefill"] for c in chunks if c.get("object") == "lmk.prefill"]
    assert prefill == [{"processed": 0, "total": 6000, "cached": 2048},
                       {"processed": 4096, "total": 6000, "cached": 2048},
                       {"processed": 5999, "total": 6000, "cached": 2048}]
    assert all(c["choices"] == [] for c in chunks if c.get("object") == "lmk.prefill")


def test_non_streaming_response():
    srv, _, _ = serve(TOOL_TURN)
    try:
        body = json.loads(post(srv, {"model": "kitten-27b", "messages": []}).read())
    finally:
        srv.shutdown()
    choice = body["choices"][0]
    assert body["object"] == "chat.completion" and choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["reasoning_content"] == "We need one call.\n"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "file_read"
    assert "index" not in choice["message"]["tool_calls"][0]


def test_engine_request_id_is_the_callers_ref_id():
    srv, engine, _ = serve(TEXT_TURN)
    try:
        post(srv, {"model": "kitten-27b", "messages": []},
             {"X-Lmk-Purpose": "turn", "X-Lmk-Ref-Id": "t-mu8ncv51.1", "traceparent": "00-abc-def-01"}).read()
    finally:
        srv.shutdown()
    assert engine.requests[0]["request_id"] == "t-mu8ncv51.1"


def test_a_malformed_tool_call_is_passed_through_as_text_not_dropped():
    srv, _, _ = serve(["x</think>", "<tool_call>BROKEN</tool_call>"])
    try:
        chunks = stream_chunks(post(srv, {"model": "kitten-27b", "stream": True, "messages": []}))
    finally:
        srv.shutdown()
    assert deltas(chunks, "content") == "<tool_call>BROKEN</tool_call>"
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "stop"


def test_max_tokens_reached_reports_length():
    srv, _, _ = serve(["still thinking"], stats=GenerationStats(prompt_tokens=10, completion_tokens=16))
    try:
        body = json.loads(post(srv, {"model": "kitten-27b", "max_tokens": 16, "messages": []}).read())
    finally:
        srv.shutdown()
    assert body["choices"][0]["finish_reason"] == "length"


def test_only_the_resident_model_is_served():
    srv, _, _ = serve(TEXT_TURN)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(srv, {"model": "some-other-model", "messages": []})
    finally:
        srv.shutdown()
    err = json.loads(e.value.read())["error"]
    assert e.value.code == 404 and err["type"] == "model_not_found" and "kitten-27b" in err["message"]
