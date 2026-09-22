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


def post_path(srv, path, body, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{srv.port}{path}", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


# WISH-019: warm a prefix without generating an answer.
def test_warmup_prefills_with_one_token_and_reports_hits():
    srv, engine, fmt = serve(["x"], stats=GenerationStats(prompt_tokens=11172, cached_tokens=0, completion_tokens=1))
    try:
        out = post_path(srv, "/lmk/v1/warmup", {"model": "kitten-27b", "tools": [{"type": "function"}],
                                                "messages": [{"role": "system", "content": "SYS"}]},
                        {"X-Lmk-Ref-Id": "warm-1"})
    finally:
        srv.shutdown()
    assert out["prompt_tokens"] == 11172 and out["cached_tokens"] == 0 and out["total_ms"] >= 0
    assert engine.requests[0]["max_tokens"] == 1 and engine.requests[0]["request_id"] == "warm-1"
    messages, tools = fmt.rendered[0]
    assert messages == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "."}]
    assert tools == [{"type": "function"}]


def test_warmup_keeps_a_closing_user_message_as_sent():
    srv, _, fmt = serve(["x"])
    try:
        post_path(srv, "/lmk/v1/warmup", {"model": "kitten-27b", "messages": [{"role": "user", "content": "hello"}]})
    finally:
        srv.shutdown()
    assert fmt.rendered[0][0] == [{"role": "user", "content": "hello"}]


IMAGE_MSG = [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]


def test_images_reach_the_engine_and_the_template_sees_placeholders():
    fmt = FakeChatFormat()
    engine = FakeEngine(MODEL, chat_format=fmt, script=["x</think>", "a cat"], modalities=["text", "image"])
    srv = LmkServer(engine, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        body = json.loads(post(srv, {"model": "kitten-27b", "messages": IMAGE_MSG}).read())
    finally:
        srv.shutdown()
    assert body["choices"][0]["message"]["content"] == "a cat"
    assert engine.requests[0]["images_b64"] == ["AAAA"]
    assert fmt.rendered[0][0][0]["content"] == [{"type": "text", "text": "what is this?"}, {"type": "image"}]


@pytest.mark.parametrize("modalities,messages,needle", [
    (["text"], IMAGE_MSG, "does not take images"),
    (["text", "image"], [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}], "inline data:"),
])
def test_bad_image_requests_are_a_400_before_any_stream_starts(modalities, messages, needle):
    engine = FakeEngine(MODEL, chat_format=FakeChatFormat(), script=["x"], modalities=modalities)
    srv = LmkServer(engine, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(srv, {"model": "kitten-27b", "stream": True, "messages": messages})
    finally:
        srv.shutdown()
    assert e.value.code == 400 and needle in json.loads(e.value.read())["error"]["message"]
    assert engine.requests == []


def serve_with_defaults(script, defaults):
    engine = FakeEngine(MODEL, chat_format=FakeChatFormat(), script=script,
                        stats=GenerationStats(prompt_tokens=441, cached_tokens=256, completion_tokens=88),
                        sampling_defaults=defaults)
    srv = LmkServer(engine, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, engine


def test_sampling_parameters_reach_the_engine_under_its_own_names_over_the_models_defaults():
    srv, engine = serve_with_defaults(TEXT_TURN, {"temp": 1.0, "top_p": 0.95, "top_k": 20})
    try:
        post(srv, {"model": "kitten-27b", "messages": [], "temperature": 0.3, "stop": ["\n\n", "END"]}).read()
        post(srv, {"model": "kitten-27b", "messages": []}).read()
    finally:
        srv.shutdown()
    # stop is lmk's to enforce (answer part only); the engine never sees it
    assert engine.requests[0]["sampling"] == {"temp": 0.3, "top_p": 0.95, "top_k": 20}
    assert engine.requests[1]["sampling"] == {"temp": 1.0, "top_p": 0.95, "top_k": 20}


def test_an_out_of_range_sampling_value_is_a_400_that_names_the_field():
    srv, engine = serve_with_defaults(TEXT_TURN, {})
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(srv, {"model": "kitten-27b", "messages": [], "top_p": 3})
    finally:
        srv.shutdown()
    err = json.loads(e.value.read())["error"]
    assert e.value.code == 400 and err["param"] == "top_p" and "between 0 (exclusive) and 1" in err["message"]
    assert engine.requests == []


def test_seed_is_accepted_but_logged_as_ignored(capsys):
    srv, engine = serve_with_defaults(TEXT_TURN, {})
    try:
        post(srv, {"model": "kitten-27b", "messages": [], "seed": 7}).read()
    finally:
        srv.shutdown()
    assert "sampling" in engine.requests[0] and "seed" not in engine.requests[0]["sampling"]
    logged = [json.loads(l) for l in capsys.readouterr().err.splitlines() if l.startswith("{")]
    ignored = [l for l in logged if l["event"] == "LmkParamIgnored"]
    assert ignored and ignored[0]["params"] == ["seed"]
    done = [l for l in logged if l["event"] == "LmkChatDone"]
    assert done[0]["sampling"] == {}


def test_stop_applies_to_the_answer_not_the_thinking_and_ends_generation():
    # "Both" appears in the thinking; it must not stop there. "found" is in the answer: stop before it.
    srv, engine = serve_with_defaults(TEXT_TURN, {})
    try:
        chunks = stream_chunks(post(srv, {"model": "kitten-27b", "messages": [], "stream": True, "stop": ["Both", " found"]}))
    finally:
        srv.shutdown()
    assert deltas(chunks, "reasoning_content") == "Both are done.\n"
    assert deltas(chunks, "content") == "Here's what I"
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]] == ["stop"]
