"""The queue as a client meets it: what comes back, and what `lmk status` shows meanwhile."""
import json
import threading
import time
import urllib.error

import pytest

from lmk.admission import Admission
from lmk.engine import FakeEngine, Generation, GenerationStats
from lmk.memory import MemoryReading, get_current_memory, set_current_memory
from lmk.server import LmkServer
from test_chat_endpoint import MODEL, FakeChatFormat, post


class HeldEngine(FakeEngine):
    """Answers only when the test says so, so one request can be 'writing' while others arrive."""

    def __init__(self, **kw):
        super().__init__(MODEL, chat_format=FakeChatFormat(), script=["</think>", "ok"], **kw)
        self.release = threading.Event()

    def generate(self, prompt_text, **kw):
        inner = super().generate(prompt_text, **kw)

        def pieces():
            self.release.wait(10)
            yield from inner.pieces

        return Generation(pieces=pieces(), stats=inner.stats)


@pytest.fixture
def memory():
    class Fake:
        pressure, free_percent = "normal", 90

        def read(self):
            return MemoryReading(self.pressure, self.free_percent, 96 * 1024**3)

    before, fake = get_current_memory(), Fake()
    set_current_memory(fake)
    yield fake
    set_current_memory(before)


def serve(engine, max_parallel=1, max_queue=16, max_wait_seconds=30):
    srv = LmkServer(engine, "127.0.0.1", 0,
                    admission=Admission(max_parallel, max_queue, max_wait_seconds, tick_seconds=0.02))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def post_in_background(srv, ref_id):
    box = {}

    def run():
        try:
            box["body"] = json.loads(post(srv, {"model": "kitten-27b", "messages": [{"role": "user", "content": "hi"}]},
                                          {"X-Lmk-Purpose": "turn", "X-Lmk-Ref-Id": ref_id}).read())
        except urllib.error.HTTPError as e:
            box["status"], box["error"] = e.code, json.loads(e.read())["error"]

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.15)
    return t, box


def test_a_request_waits_its_turn_is_visible_while_it_waits_and_then_answers(memory):
    engine = HeldEngine(stats=GenerationStats(prompt_tokens=100, cached_tokens=90, completion_tokens=1), gpu_bytes=5)
    srv = serve(engine)
    first, first_box = post_in_background(srv, "s/first")
    second, second_box = post_in_background(srv, "s/second")

    status = srv.status()
    assert [r["ref_id"] for r in status["in_flight"]] == ["s/first"]
    assert [(w["ref_id"], w["reason"]) for w in status["waiting"]] == \
        [("s/second", "1 requests are being answered (requests.max_parallel)")]
    assert status["memory"] == {"pressure": "normal", "free_percent": 90, "total_bytes": 96 * 1024**3,
                                "lmk_gpu_bytes": 5, "lmk_gpu_peak_in_use_bytes": 5}
    assert status["requests"] == {"answering": 1, "max_parallel": 1, "waiting": 1, "max_queue": 16,
                                  "tokens_in_memory": 200000, "token_budget": None}

    engine.release.set()
    first.join(5), second.join(5)
    assert first_box["body"]["choices"][0]["message"]["content"] == "ok"
    assert second_box["body"]["choices"][0]["message"]["content"] == "ok"
    done = srv.status()
    assert done["waiting"] == [] and done["in_flight"] == []
    assert done["totals"] == {"answered": 2, "refused": 0, "failed": 0, "cancelled": 0,
                              "prompt_tokens": 200, "cached_tokens": 180}
    assert [(f["ref_id"], f["outcome"], f["prompt_tokens"], f["cached_tokens"]) for f in done["recent"]] == \
        [("s/second", "stop", 100, 90), ("s/first", "stop", 100, 90)]      # newest first


def test_one_that_could_not_start_in_time_gets_a_503_that_says_why(memory):
    engine = HeldEngine(stats=GenerationStats(prompt_tokens=100, cached_tokens=90, completion_tokens=1))
    srv = serve(engine, max_wait_seconds=1)
    first, _ = post_in_background(srv, "s/first")
    late, late_box = post_in_background(srv, "s/late")
    late.join(5)
    assert late_box["status"] == 503 and late_box["error"]["type"] == "waited_too_long"
    assert late_box["error"]["message"] == ("lmk waited 1s and could not start: 1 requests are being answered "
                                            "(requests.max_parallel)")
    assert srv.status()["totals"]["refused"] == 1
    engine.release.set()
    first.join(5)


def test_a_full_queue_answers_503_at_once(memory):
    engine = HeldEngine(stats=GenerationStats(prompt_tokens=100, cached_tokens=90, completion_tokens=1))
    srv = serve(engine, max_queue=1)
    post_in_background(srv, "s/first")
    post_in_background(srv, "s/waiting")
    refused, box = post_in_background(srv, "s/refused")
    refused.join(5)
    assert box["status"] == 503 and box["error"]["type"] == "queue_full"
    assert box["error"]["message"] == "lmk is busy: 1 requests are already waiting. Try again shortly."
    engine.release.set()


def test_the_place_is_given_back_even_when_the_request_blows_up(memory):
    class Exploding(HeldEngine):
        def generate(self, prompt_text, **kw):
            raise RuntimeError("engine fell over")

    srv = serve(Exploding(stats=GenerationStats(prompt_tokens=10, cached_tokens=0, completion_tokens=0)))
    for ref in ("s/one", "s/two"):  # the second would wait forever if the first kept its place
        t, box = post_in_background(srv, ref)
        t.join(5)
        assert box["status"] == 500 and box["error"]["type"] == "internal_error"
        assert "engine fell over" in box["error"]["message"]  # told, not just disconnected
    after = srv.status()
    assert after["in_flight"] == [] and after["waiting"] == []
    assert after["totals"]["failed"] == 2 and [f["outcome"] for f in after["recent"]] == ["failed", "failed"]


# Warmup is the lowest priority (kitten design 2026-09-25-prewarm): it yields to a request
# at the next prefill step and says so; one that never sees an idle engine says that too.
class SteppedEngine(FakeEngine):
    """Each prefill step waits for the test to let it through."""

    def __init__(self, steps, **kw):
        super().__init__(MODEL, chat_format=FakeChatFormat(), script=["</think>", "ok"], **kw)
        self.step = threading.Semaphore(0)
        self._steps = steps

    def generate(self, prompt_text, *, on_prefill, **kw):
        inner = super().generate(prompt_text, on_prefill=lambda *_: True, **kw)

        def pieces():
            for processed in self._steps:
                self.step.acquire(timeout=10)
                if not on_prefill(processed, inner.stats.prompt_tokens, inner.stats.cached_tokens):
                    return
            yield from inner.pieces

        return Generation(pieces=pieces(), stats=inner.stats)


def warmup_in_background(srv, ref_id):
    from test_chat_endpoint import post_path
    box = {}

    def run():
        box["body"] = post_path(srv, "/lmk/v1/warmup", {"model": "kitten-27b",
                                                        "messages": [{"role": "system", "content": "SYS"}]},
                                {"X-Lmk-Purpose": "warmup", "X-Lmk-Ref-Id": ref_id})

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.15)
    return t, box


def test_a_warmup_yields_to_a_request_at_the_next_prefill_step(memory):
    engine = SteppedEngine([2048, 4096, 6144],
                           stats=GenerationStats(prompt_tokens=30000, cached_tokens=0, completion_tokens=1))
    srv = serve(engine, max_parallel=2)
    try:
        warm, warm_box = warmup_in_background(srv, "s/warm")
        engine.step.release()                                    # the warmup reads its first step
        time.sleep(0.1)
        turn, turn_box = post_in_background(srv, "s/turn")       # a request arrives
        engine.step.release(10)                                  # let every step through from here on
        warm.join(5), turn.join(5)
    finally:
        srv.shutdown()
    assert warm_box["body"]["outcome"] == "yielded"
    assert turn_box["body"]["choices"][0]["message"]["content"] == "ok"
    recent = srv.status()["recent"]
    assert [(f["ref_id"], f["outcome"]) for f in recent] == [("s/turn", "stop"), ("s/warm", "yielded")]


def test_a_warmup_that_never_sees_an_idle_engine_answers_not_started(memory):
    engine = HeldEngine(stats=GenerationStats(prompt_tokens=100, cached_tokens=90, completion_tokens=1))
    srv = LmkServer(engine, "127.0.0.1", 0,
                    admission=Admission(1, 16, 30, tick_seconds=0.02, warmup_wait_seconds=1))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        busy, _ = post_in_background(srv, "s/busy")
        from test_chat_endpoint import post_path
        out = post_path(srv, "/lmk/v1/warmup", {"model": "kitten-27b", "messages": [{"role": "system", "content": "SYS"}]},
                        {"X-Lmk-Purpose": "warmup", "X-Lmk-Ref-Id": "s/warm"})
        engine.release.set()
        busy.join(5)
    finally:
        srv.shutdown()
    assert out == {"outcome": "not_started", "reason": "a warmup waits for an idle engine"}
