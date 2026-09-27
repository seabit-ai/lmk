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
    think_open, think_close = "<think>", "</think>"

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
    assert out["outcome"] == "done"
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


class CountingSeeds:
    def __init__(self, first=1000):
        self.next = first

    def draw(self):
        self.next += 1
        return self.next - 1


@pytest.fixture
def seeds():
    from lmk.sampling import get_current_seed_source, set_current_seed_source
    before, fake = get_current_seed_source(), CountingSeeds()
    set_current_seed_source(fake)
    yield fake
    set_current_seed_source(before)


def test_sampling_parameters_reach_the_engine_under_its_own_names_over_the_models_defaults(seeds):
    srv, engine = serve_with_defaults(TEXT_TURN, {"temp": 1.0, "top_p": 0.95, "top_k": 20})
    try:
        post(srv, {"model": "kitten-27b", "messages": [], "temperature": 0.3, "stop": ["\n\n", "END"]}).read()
        post(srv, {"model": "kitten-27b", "messages": []}).read()
    finally:
        srv.shutdown()
    # stop is lmk's to enforce (answer part only); the engine never sees it
    assert engine.requests[0]["sampling"] == {"temp": 0.3, "top_p": 0.95, "top_k": 20, "seed": 1000}
    assert engine.requests[1]["sampling"] == {"temp": 1.0, "top_p": 0.95, "top_k": 20, "seed": 1001}


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


def _logged(capsys, event):
    return [l for l in (json.loads(l) for l in capsys.readouterr().err.splitlines() if l.startswith("{"))
            if l["event"] == event]


@pytest.mark.parametrize("seed", [7, 2**64 - 1])
def test_a_requests_seed_reaches_the_engine_comes_back_and_is_logged(capsys, seeds, seed):
    srv, engine = serve_with_defaults(TEXT_TURN, {"temp": 1.0})
    try:
        chunks = stream_chunks(post(srv, {"model": "kitten-27b", "messages": [], "seed": seed, "stream": True}))
    finally:
        srv.shutdown()
    assert engine.requests[0]["sampling"] == {"temp": 1.0, "seed": seed}
    assert [c["lmk"]["seed"] for c in chunks if "usage" in c] == [seed]
    done = _logged(capsys, "LmkChatDone")[0]
    assert (done["seed"], done["seedFrom"], done["sampling"]) == (seed, "request", {"temp": 1.0})
    assert seeds.next == 1000   # nothing drawn


@pytest.mark.parametrize("defaults, body", [({}, {}), ({"temp": 1.0}, {"temperature": 0}),
                                            ({"temp": 1.0}, {"temperature": 0, "seed": 5})])
def test_a_greedy_answer_has_no_seed(capsys, seeds, defaults, body):
    srv, engine = serve_with_defaults(TEXT_TURN, defaults)
    try:
        answer = json.loads(post(srv, {"model": "kitten-27b", "messages": [], **body}).read())
    finally:
        srv.shutdown()
    assert "seed" not in engine.requests[0]["sampling"] and answer["lmk"]["seed"] is None
    done = _logged(capsys, "LmkChatDone")[0]
    assert (done["seed"], done["seedFrom"]) == (None, "greedy") and seeds.next == 1000


def test_without_a_seed_lmk_draws_one_and_says_so(capsys, seeds):
    srv, engine = serve_with_defaults(TEXT_TURN, {"temp": 1.0})
    try:
        body = json.loads(post(srv, {"model": "kitten-27b", "messages": []}).read())
    finally:
        srv.shutdown()
    assert engine.requests[0]["sampling"] == {"temp": 1.0, "seed": 1000}
    assert body["lmk"]["seed"] == 1000          # the non-streamed answer carries lmk's fields too
    logged = capsys.readouterr().err
    done = [json.loads(l) for l in logged.splitlines() if '"LmkChatDone"' in l][0]
    assert (done["seed"], done["seedFrom"], done["sampling"]) == (1000, "lmk", {"temp": 1.0})
    assert '"LmkParamIgnored"' not in logged


def test_a_warmup_draws_no_seed(seeds):
    srv, engine, _ = serve(["x"])
    try:
        post_path(srv, "/lmk/v1/warmup", {"model": "kitten-27b", "messages": [{"role": "user", "content": "hi"}]})
    finally:
        srv.shutdown()
    assert engine.requests[0]["sampling"] is None and seeds.next == 1000


@pytest.mark.parametrize("seed", ["abc", -1, 2**64, 1.5])
def test_a_seed_outside_uint64_is_a_400_that_says_the_range(seeds, seed):
    srv, engine = serve_with_defaults(TEXT_TURN, {"temp": 1.0})
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(srv, {"model": "kitten-27b", "messages": [], "seed": seed})
    finally:
        srv.shutdown()
    err = json.loads(e.value.read())["error"]
    assert e.value.code == 400 and err["param"] == "seed" and "0 to 18446744073709551615" in err["message"]
    assert engine.requests == []


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


class SteppingClock:
    """Every reading moves time on by `step_ms` — a decode that takes real time, without sleeping."""

    def __init__(self, step_ms):
        self.now, self.step_ms = 1_000_000, step_ms

    def mono_ms(self):
        self.now += self.step_ms
        return self.now

    def wall_ms(self):
        return self.now


def decode_chunks(step_ms):
    from lmk.clock import get_current_clock, set_current_clock
    before = get_current_clock()
    set_current_clock(SteppingClock(step_ms))
    try:
        srv, _, _ = serve(TEXT_TURN, stats=GenerationStats(prompt_tokens=441, cached_tokens=256, completion_tokens=88))
        try:
            chunks = stream_chunks(post(srv, {"model": "kitten-27b", "stream": True, "messages": []}))
        finally:
            srv.shutdown()
    finally:
        set_current_clock(before)
    return [c for c in chunks if c.get("object") == "lmk.decode"]


# kitten design 2026-09-24-llm-progress §9: decode progress rides the stream like prefill
# progress does — once when the first token comes out, then at most once a second.
def test_decode_progress_rides_the_stream_once_a_second():
    frozen = decode_chunks(step_ms=0)
    assert len(frozen) == 1, "no time passes: only the first-token report"
    assert frozen[0]["choices"] == []
    first = frozen[0]["lmk"]["decode"]
    assert first["part"] == "thinking" and first["completion_tokens"] == 88

    moving = decode_chunks(step_ms=600)
    assert 1 < len(moving) <= len(TEXT_TURN)
    assert moving[-1]["lmk"]["decode"]["part"] == "answering"
    assert moving[-1]["lmk"]["decode"]["tokens_per_s"] > 0


def test_stopping_early_closes_the_engines_generator(monkeypatch):
    # the fork's generator takes the row out of the batch when closed; MlxEngine must close it, not drop it
    import sys
    import types

    from lmk.engine import MlxEngine

    closed = []

    held = []   # a reference elsewhere (as the engine's own bookkeeping may keep): garbage collection won't close it

    def create_generator(kit, tokens, **kwargs):
        def gen():
            try:
                for piece in ["a", "b", "c"]:
                    yield types.SimpleNamespace(text=piece, tokens=[1])
            finally:
                closed.append(kwargs["request_id"])
        held.append(gen())
        return held[-1]

    monkeypatch.setitem(sys.modules, "mlx_engine.generate",
                        types.SimpleNamespace(create_generator=create_generator, tokenize=lambda kit, text: [1]))
    monkeypatch.setitem(sys.modules, "mlx_engine.utils.prompt_progress_reporter",
                        types.SimpleNamespace(PromptProgressReporter=object))
    engine = object.__new__(MlxEngine)
    engine._kit, engine._draft_tokens = types.SimpleNamespace(), None
    generation = engine.generate("p", max_tokens=None, request_id="r-1", on_prefill=lambda *a: True, tokens=[1])
    assert next(iter(generation)) == "a"
    generation.pieces.close()
    assert closed == ["r-1"]
