"""What `lmk status` knows about requests: the ones running, the last few that finished,
and the totals since start. One object, written by request threads, read by status.

States of a running request, in order:
  starting  admitted; the prompt's cached part is being brought back from disk
  prefill   the uncached part of the prompt is being read (with progress)
  decode    tokens are being produced. `part` — thinking / answering / tool call — is NOT
            something the engine knows: to the model it is all tokens. lmk reads it off the
            markers in the text (OutputSplitter), the same way it splits the API output.
"""
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

from lmk.clock import get_current_clock

RECENT_KEPT = 5


@dataclass(eq=False)
class Running:
    purpose: Optional[str]
    ref_id: Optional[str]
    traceparent: Optional[str]
    prompt_tokens: int
    started_mono_ms: int
    state: str = "starting"
    part: Optional[str] = None
    cached_tokens: Optional[int] = None          # known once the engine has looked
    prefill: Optional[dict] = None               # {"processed", "total", "cached"}
    completion_tokens: int = 0
    first_token_mono_ms: Optional[int] = None

    def on_prefill(self, progress: dict) -> None:
        self.state, self.prefill = "prefill", progress
        self.cached_tokens = progress["cached"]
        self.prompt_tokens = max(self.prompt_tokens, progress["total"])

    def on_decode(self, part: str, completion_tokens: int) -> None:
        if self.first_token_mono_ms is None:
            self.first_token_mono_ms = get_current_clock().mono_ms()
        self.state, self.part, self.completion_tokens = "decode", part, completion_tokens

    def decode_tokens_per_s(self, now_ms: int) -> Optional[float]:
        if self.first_token_mono_ms is None or now_ms <= self.first_token_mono_ms or self.completion_tokens < 2:
            return None
        return round(self.completion_tokens / ((now_ms - self.first_token_mono_ms) / 1000), 1)


@dataclass
class Totals:
    answered: int = 0
    refused: int = 0      # 503: the queue was full, or the wait ran out
    failed: int = 0       # 500: lmk or the engine fell over
    cancelled: int = 0    # the client went away
    prompt_tokens: int = 0
    cached_tokens: int = 0


class Board:
    def __init__(self):
        self._lock = threading.Lock()
        self._running: list[Running] = []
        self._recent: deque[dict] = deque(maxlen=RECENT_KEPT)
        self._totals = Totals()

    def begin(self, purpose, ref_id, traceparent, prompt_tokens: int) -> Running:
        r = Running(purpose=purpose, ref_id=ref_id, traceparent=traceparent, prompt_tokens=prompt_tokens,
                    started_mono_ms=get_current_clock().mono_ms())
        with self._lock:
            self._running.append(r)
        return r

    def refused(self) -> None:
        with self._lock:
            self._totals.refused += 1

    def finish(self, r: Running, outcome: str, prompt_tokens: Optional[int] = None,
               cached_tokens: Optional[int] = None, completion_tokens: Optional[int] = None) -> None:
        """outcome: stop | tool call | length | warmed | cancelled | failed"""
        now = get_current_clock().mono_ms()
        if prompt_tokens is not None:
            r.prompt_tokens, r.cached_tokens = prompt_tokens, cached_tokens
        if completion_tokens is not None:
            r.completion_tokens = completion_tokens
        with self._lock:
            if r in self._running:
                self._running.remove(r)
            if outcome == "failed":
                self._totals.failed += 1
            elif outcome == "cancelled":
                self._totals.cancelled += 1
            else:
                self._totals.answered += 1
            if outcome != "failed" and r.cached_tokens is not None:
                # the denominator next to the numerator: whether the cache earns its disk is a ratio
                self._totals.prompt_tokens += r.prompt_tokens
                self._totals.cached_tokens += r.cached_tokens
            self._recent.appendleft({
                "purpose": r.purpose, "ref_id": r.ref_id, "outcome": outcome,
                "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens,
                "first_token_ms": None if r.first_token_mono_ms is None else r.first_token_mono_ms - r.started_mono_ms,
                "completion_tokens": r.completion_tokens, "decode_tokens_per_s": r.decode_tokens_per_s(now),
                "total_ms": now - r.started_mono_ms, "finished_mono_ms": now})

    def snapshot(self) -> dict:
        now = get_current_clock().mono_ms()
        with self._lock:
            running = [{"purpose": r.purpose, "ref_id": r.ref_id, "traceparent": r.traceparent,
                        "state": r.state, "part": r.part, "prompt_tokens": r.prompt_tokens,
                        "cached_tokens": r.cached_tokens, "prefill": r.prefill,
                        "completion_tokens": r.completion_tokens,
                        "decode_tokens_per_s": r.decode_tokens_per_s(now),
                        "running_ms": now - r.started_mono_ms} for r in self._running]
            recent = [{**{k: v for k, v in f.items() if k != "finished_mono_ms"},
                       "ago_ms": now - f["finished_mono_ms"]} for f in self._recent]
            totals = dict(vars(self._totals))
        return {"in_flight": running, "recent": recent, "totals": totals}
