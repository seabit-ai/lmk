"""One queue in front of the engine, first come first served (design memory-guard §F).

The request at the head passes four rules, in order, or stays at the head — and
everyone behind it waits too:
  1. fewer than max_parallel requests are being answered
  2. nobody is being answered, or this request has little new prompt to read
     (reading a long prompt stalls every request that is writing — research MG-006)
  3. its tokens fit next to the ones already admitted
  4. this Mac is not critically short of memory
Nothing is sent to the client before admission, so a request that waited too long
gets an honest 503.
"""
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from lmk import log
from lmk.clock import get_current_clock
from lmk.memory import get_current_memory

# One prefill step of the engine (DEFAULT_PREFILL_STEP_SIZE): a prompt whose uncached
# part fits in it is read in a single scheduler turn and stalls nobody.
SHORT_PROMPT_TOKENS = 2048


class QueueFull(Exception):
    def __init__(self, waiting: int):
        super().__init__(f"{waiting} requests are already waiting")
        self.waiting = waiting


class WaitedTooLong(Exception):
    def __init__(self, waited_s: int, reason: str):
        super().__init__(f"waited {waited_s}s and could not start: {reason}")
        self.waited_s = waited_s
        self.reason = reason


@dataclass(eq=False)
class Ticket:
    purpose: Optional[str]
    ref_id: Optional[str]
    tokens: int                        # prompt + what it may generate: its share of the KV memory
    uncached_tokens: Optional[int]     # None: unknown — treated as long
    entered_mono_ms: int = 0
    reason: str = ""
    admitted: bool = field(default=False)


class Admission:
    def __init__(self, max_parallel: int, max_queue: int, max_wait_seconds: int,
                 token_budget: Optional[int] = None, tick_seconds: float = 1.0):
        self._max_parallel = max_parallel
        self._max_queue = max_queue
        self._max_wait_ms = max_wait_seconds * 1000
        self._token_budget = token_budget    # None: the engine reported no memory fit; rule 3 is off
        self._tick_seconds = tick_seconds    # rule 4 changes with the outside world: re-read while anyone waits
        self._cond = threading.Condition()
        self._queue: deque[Ticket] = deque()
        self._admitted: list[Ticket] = []
        self._was_critical = False
        self._critical_since_ms = 0

    def _blocked(self, t: Ticket) -> Optional[str]:
        if len(self._admitted) >= self._max_parallel:
            return f"{len(self._admitted)} requests are being answered (requests.max_parallel)"
        if self._admitted and (t.uncached_tokens is None or t.uncached_tokens > SHORT_PROMPT_TOKENS):
            new = "a long" if t.uncached_tokens is None else f"{t.uncached_tokens:,} tokens of"
            return f"another request is being answered, and this one has {new} new prompt to read first"
        if self._admitted and self._token_budget is not None:
            in_use = sum(a.tokens for a in self._admitted)
            if in_use + t.tokens > self._token_budget:
                return (f"not enough memory for both: {in_use:,} tokens in use, this one needs {t.tokens:,}, "
                        f"{self._token_budget:,} fit")
        memory = get_current_memory().read()
        self._note_pressure(memory.critical, memory.free_percent)
        if memory.critical:
            return f"this Mac is critically short of memory ({memory.free_percent}% free)"
        return None

    def _note_pressure(self, critical: bool, free_percent: int) -> None:
        now = get_current_clock().mono_ms()
        if critical and not self._was_critical:
            self._critical_since_ms = now
            log.warn("LmkMemoryCritical", "memory pressure is critical; new requests wait", freePercent=free_percent)
        elif self._was_critical and not critical:
            log.info("LmkMemoryRecovered", "memory pressure is no longer critical",
                     lastedMs=now - self._critical_since_ms, freePercent=free_percent)
        self._was_critical = critical

    def enter(self, ticket: Ticket) -> None:
        clock = get_current_clock()
        with self._cond:
            if len(self._queue) >= self._max_queue:
                raise QueueFull(len(self._queue))
            ticket.entered_mono_ms = clock.mono_ms()
            self._queue.append(ticket)
            try:
                while True:
                    if self._queue[0] is ticket:
                        reason = self._blocked(ticket)
                        if reason is None:
                            self._queue.popleft()
                            ticket.admitted = True
                            ticket.reason = ""
                            self._admitted.append(ticket)
                            self._cond.notify_all()  # the next in line may fit too
                            return
                        ticket.reason = reason
                    else:
                        ticket.reason = "requests ahead of it are waiting"
                    waited_ms = clock.mono_ms() - ticket.entered_mono_ms
                    if waited_ms >= self._max_wait_ms:
                        raise WaitedTooLong(waited_ms // 1000, ticket.reason)
                    self._cond.wait(timeout=min(self._tick_seconds, (self._max_wait_ms - waited_ms) / 1000))
            except BaseException:
                if ticket in self._queue:
                    self._queue.remove(ticket)
                self._cond.notify_all()
                raise

    def leave(self, ticket: Ticket) -> None:
        with self._cond:
            if ticket in self._admitted:
                self._admitted.remove(ticket)
            self._cond.notify_all()

    def counts(self) -> dict:
        with self._cond:
            return {"answering": len(self._admitted), "max_parallel": self._max_parallel,
                    "waiting": len(self._queue), "max_queue": self._max_queue,
                    "tokens_in_memory": sum(a.tokens for a in self._admitted), "token_budget": self._token_budget}

    def waiting(self) -> list[dict]:
        now = get_current_clock().mono_ms()
        with self._cond:
            return [{"purpose": t.purpose, "ref_id": t.ref_id, "reason": t.reason,
                     "waited_ms": now - t.entered_mono_ms} for t in self._queue]
