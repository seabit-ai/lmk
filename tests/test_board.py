from lmk.board import RECENT_KEPT, Board
from lmk.clock import get_current_clock, set_current_clock


class TickingClock:
    def __init__(self):
        self.now = 1_000

    def mono_ms(self):
        return self.now

    def wall_ms(self):
        return self.now


def test_a_request_moves_through_starting_prefill_decode_and_then_into_the_recent_list():
    before, clock = get_current_clock(), TickingClock()
    set_current_clock(clock)
    try:
        board = Board()
        r = board.begin("turn", "s/1", None, prompt_tokens=27_190)
        assert board.snapshot()["in_flight"][0]["state"] == "starting"

        clock.now += 900
        r.on_prefill({"processed": 27_136, "total": 27_190, "cached": 27_136})
        snap = board.snapshot()["in_flight"][0]
        assert (snap["state"], snap["cached_tokens"], snap["running_ms"]) == ("prefill", 27_136, 900)

        clock.now += 100
        r.on_decode("thinking", 1)
        clock.now += 3_000
        r.on_decode("answering", 100)
        snap = board.snapshot()["in_flight"][0]
        assert (snap["state"], snap["part"], snap["completion_tokens"]) == ("decode", "answering", 100)
        assert snap["decode_tokens_per_s"] == 33.3        # 100 tokens over the 3s since the first one

        board.finish(r, "stop", prompt_tokens=27_190, cached_tokens=27_136, completion_tokens=100)
        clock.now += 12_000
        done = board.snapshot()
        assert done["in_flight"] == []
        assert done["recent"][0] == {"purpose": "turn", "ref_id": "s/1", "outcome": "stop", "prompt_tokens": 27_190,
                                     "cached_tokens": 27_136, "first_token_ms": 1_000, "completion_tokens": 100,
                                     "decode_tokens_per_s": 33.3, "total_ms": 4_000, "ago_ms": 12_000}
        assert done["totals"] == {"answered": 1, "refused": 0, "failed": 0, "cancelled": 0,
                                  "prompt_tokens": 27_190, "cached_tokens": 27_136}
    finally:
        set_current_clock(before)


def test_only_the_last_few_finished_requests_are_kept_and_failures_do_not_count_as_cache_traffic():
    board = Board()
    for i in range(RECENT_KEPT + 3):
        board.finish(board.begin("turn", f"s/{i}", None, 100), "stop", prompt_tokens=100, cached_tokens=50)
    board.finish(board.begin("turn", "s/boom", None, 100), "failed")
    board.finish(board.begin("turn", "s/gone", None, 100), "cancelled")
    board.refused()
    snap = board.snapshot()
    assert len(snap["recent"]) == RECENT_KEPT and snap["recent"][0]["ref_id"] == "s/gone"
    assert snap["totals"] == {"answered": RECENT_KEPT + 3, "refused": 1, "failed": 1, "cancelled": 1,
                              "prompt_tokens": 100 * (RECENT_KEPT + 3), "cached_tokens": 50 * (RECENT_KEPT + 3)}
