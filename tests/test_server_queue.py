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
                                "lmk_gpu_bytes": 5}

    engine.release.set()
    first.join(5), second.join(5)
    assert first_box["body"]["choices"][0]["message"]["content"] == "ok"
    assert second_box["body"]["choices"][0]["message"]["content"] == "ok"
    assert srv.status()["waiting"] == [] and srv.status()["in_flight"] == []


def test_one_that_could_not_start_in_time_gets_a_503_that_says_why(memory):
    engine = HeldEngine(stats=GenerationStats(prompt_tokens=100, cached_tokens=90, completion_tokens=1))
    srv = serve(engine, max_wait_seconds=1)
    first, _ = post_in_background(srv, "s/first")
    late, late_box = post_in_background(srv, "s/late")
    late.join(5)
    assert late_box["status"] == 503 and late_box["error"]["type"] == "waited_too_long"
    assert late_box["error"]["message"] == ("lmk waited 1s and could not start: 1 requests are being answered "
                                            "(requests.max_parallel)")
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
    assert srv.status()["in_flight"] == [] and srv.status()["waiting"] == []
