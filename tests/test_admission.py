import threading
import time

import pytest

from lmk.admission import Admission, QueueFull, Ticket, WaitedTooLong
from lmk.memory import MemoryReading, get_current_memory, set_current_memory

GB = 1024**3


class FakeMemory:
    def __init__(self):
        self.pressure, self.free_percent = "normal", 80

    def read(self):
        return MemoryReading(self.pressure, self.free_percent, 96 * GB)


@pytest.fixture
def memory():
    before, fake = get_current_memory(), FakeMemory()
    set_current_memory(fake)
    yield fake
    set_current_memory(before)


def ticket(name, tokens=1000, uncached=10):
    return Ticket(purpose="turn", ref_id=name, tokens=tokens, uncached_tokens=uncached)


class Entering:
    """enter() on its own thread, so a test can look at a request while it waits."""

    def __init__(self, admission, t):
        self.ticket, self.error = t, None
        self._thread = threading.Thread(target=self._run, args=(admission,), daemon=True)
        self._thread.start()

    def _run(self, admission):
        try:
            admission.enter(self.ticket)
        except Exception as e:  # noqa: BLE001 - handed to the test
            self.error = e

    def settle(self, seconds=0.15):
        time.sleep(seconds)
        return self

    def done(self, timeout=2.0):
        self._thread.join(timeout)
        assert not self._thread.is_alive(), f"still waiting: {self.ticket.reason}"
        return self


def make(max_parallel=2, max_queue=16, max_wait_seconds=30, budget=None):
    return Admission(max_parallel, max_queue, max_wait_seconds, token_budget=budget, tick_seconds=0.02)


def test_short_requests_run_side_by_side_up_to_the_limit_then_wait_their_turn(memory):
    a = make(max_parallel=2)
    t1, t2 = ticket("one"), ticket("two")
    a.enter(t1), a.enter(t2)
    third = Entering(a, ticket("three")).settle()
    assert not third.ticket.admitted
    assert third.ticket.reason == "2 requests are being answered (requests.max_parallel)"
    assert [w["ref_id"] for w in a.waiting()] == ["three"]
    a.leave(t1)
    assert third.done().ticket.admitted and a.waiting() == []


def test_a_long_new_prompt_waits_while_someone_is_being_answered_but_not_when_alone(memory):
    a = make()
    alone = ticket("long-alone", uncached=13_441)
    a.enter(alone)                                   # nobody to stall: goes straight in
    late = Entering(a, ticket("long-late", uncached=13_441)).settle()
    assert late.ticket.reason == ("another request is being answered, and this one has 13,441 tokens of "
                                  "new prompt to read first")
    a.leave(alone)
    assert late.done().ticket.admitted


def test_the_next_step_of_a_conversation_is_never_held_up_by_its_long_cached_prompt(memory):
    a = make()
    a.enter(ticket("writing"))
    a.enter(ticket("agent-step", tokens=27_190, uncached=54))   # 27k prompt, 54 new tokens
    assert a.waiting() == []


def test_a_prompt_whose_cached_part_is_unknown_is_treated_as_long(memory):
    a = make()
    a.enter(ticket("writing"))
    unknown = Entering(a, ticket("with-image", uncached=None)).settle()
    assert "a long new prompt" in unknown.ticket.reason


def test_first_come_first_served_a_short_request_does_not_jump_a_waiting_long_one(memory):
    a = make()
    writing = ticket("writing")
    a.enter(writing)
    long_one = Entering(a, ticket("long", uncached=9000)).settle()
    short_one = Entering(a, ticket("short")).settle()
    assert short_one.ticket.reason == "requests ahead of it are waiting"
    a.leave(writing)
    assert long_one.done().ticket.admitted
    assert short_one.done().ticket.admitted           # short: fine next to the long one once it is in


def test_two_requests_that_do_not_fit_in_memory_together_take_turns(memory):
    a = make(budget=65_000)
    first = ticket("first", tokens=50_000)
    a.enter(first)
    second = Entering(a, ticket("second", tokens=40_000)).settle()
    assert second.ticket.reason == ("not enough memory for both: 50,000 tokens in use, this one needs 40,000, "
                                    "65,000 fit")
    a.leave(first)
    assert second.done().ticket.admitted


def test_a_single_request_may_use_the_whole_window_whatever_the_budget_says(memory):
    a = make(budget=65_000)
    a.enter(ticket("huge", tokens=200_000))           # alone: the engine's own context check is the judge
    assert a.waiting() == []


def test_critical_memory_holds_new_requests_until_it_passes(memory):
    a = make()
    memory.pressure, memory.free_percent = "critical", 3
    held = Entering(a, ticket("held")).settle()
    assert held.ticket.reason == "this Mac is critically short of memory (3% free)"
    memory.pressure, memory.free_percent = "warning", 14   # a warning changes nothing
    assert held.done().ticket.admitted


def test_a_request_that_cannot_start_in_time_is_refused_with_what_it_was_waiting_for(memory):
    a = Admission(1, 16, max_wait_seconds=1, tick_seconds=0.02)
    a.enter(ticket("writing"))
    late = Entering(a, ticket("late")).done(timeout=3.0)
    assert isinstance(late.error, WaitedTooLong)
    assert late.error.waited_s == 1 and "1 requests are being answered" in late.error.reason
    assert a.waiting() == []                          # it left the queue on its way out


def test_a_full_queue_refuses_the_next_request_at_once(memory):
    a = make(max_parallel=1, max_queue=2)
    a.enter(ticket("writing"))
    Entering(a, ticket("w1")).settle(0.05), Entering(a, ticket("w2")).settle(0.05)
    with pytest.raises(QueueFull, match="2 requests are already waiting"):
        a.enter(ticket("one-too-many"))
