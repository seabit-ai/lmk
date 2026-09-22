"""Streaming three-way split of raw model output: reasoning / answer text /
tool-call blocks. The engine emits plain text (research LMS-014); which markers
delimit a tool call or a thought is the model family's business and arrives from
outside (design §4, lmk.chatformat.Dialect) — this file knows no marker of its own.
"""
from dataclasses import dataclass

_WS = " \t\r\n"


@dataclass(frozen=True)
class Markers:
    tool_call_start: str | None
    tool_call_end: str | None
    think_open: str | None      # None: this family has no thinking block; everything is text or tool
    think_close: str | None


class OutputSplitter:
    def __init__(self, markers: Markers, starts_in_reasoning, on_reasoning, on_text, on_tool_block):
        self._tool_start = markers.tool_call_start
        self._tool_end = markers.tool_call_end
        self._think_open = markers.think_open
        self._think_close = markers.think_close
        self._on_reasoning = on_reasoning
        self._on_text = on_text
        self._on_tool_block = on_tool_block
        # "leading": nothing decided yet (the model may or may not open a think block itself)
        self._phase = "reasoning" if starts_in_reasoning else ("leading" if self._think_open else "text")
        self._pending = ""

    @property
    def part(self) -> str:
        """What the model is writing right now, for `lmk status`."""
        return {"reasoning": "thinking", "tool": "tool call"}.get(self._phase, "answering")

    def write(self, fragment: str) -> None:
        self._pending += fragment
        while self._step():
            pass

    def close(self) -> None:
        if self._phase == "reasoning":
            self._emit(self._on_reasoning, self._pending)
        elif self._phase in ("leading", "text"):
            self._emit(self._on_text, self._pending)
        # "gap": only whitespace was held. "tool": an unfinished call is dropped, never parsed.
        self._pending = ""

    def _step(self) -> bool:
        p = self._pending
        if self._phase == "leading":
            trimmed = p.lstrip(_WS)
            if not trimmed:
                return False
            if trimmed.startswith(self._think_open):
                self._pending, self._phase = trimmed[len(self._think_open):], "reasoning"
                return True
            if self._think_open.startswith(trimmed):
                return False
            self._phase = "text"
            return True
        if self._phase == "reasoning":
            return self._until(self._think_close, self._on_reasoning, "gap")
        if self._phase == "gap":
            self._pending = p.lstrip(_WS)
            if not self._pending:
                return False
            self._phase = "text"
            return True
        if self._phase == "text":
            # a tool call or another think block may begin anywhere in the answer: Gemma 4 opens
            # thought channels mid-turn, even a second, empty one right before a tool call (exp06)
            return self._until_any([(self._tool_start, "tool"), (self._think_open, "reasoning")], self._on_text)
        # "tool": hold everything until the end marker, then hand over the whole block
        i = p.find(self._tool_end)
        if i < 0:
            return False
        self._on_tool_block(p[:i])
        self._pending, self._phase = p[i + len(self._tool_end):], "gap"
        return True

    def _until(self, marker: str, emit, next_phase: str) -> bool:
        return self._until_any([(marker, next_phase)], emit)

    def _until_any(self, markers: list, emit) -> bool:
        """Emit up to the earliest of the markers and switch to its phase; hold back any tail that
        could be the start of one of them."""
        p = self._pending
        live = [(m, phase) for m, phase in markers if m]
        if not live:
            self._emit(emit, p)
            self._pending = ""
            return False
        hits = [(p.find(m), m, phase) for m, phase in live if m in p]
        if hits:
            i, m, phase = min(hits)
            self._emit(emit, p[:i])
            self._pending, self._phase = p[i + len(m):], phase
            return True
        keep = max(_partial_suffix_len(p, m) for m, _ in live)
        self._emit(emit, p[:len(p) - keep])
        self._pending = p[len(p) - keep:]
        return False

    @staticmethod
    def _emit(fn, text: str) -> None:
        if text:
            fn(text)


def _partial_suffix_len(s: str, marker: str) -> int:
    """Longest suffix of s that is a proper prefix of marker — the bytes to hold back."""
    for n in range(min(len(marker) - 1, len(s)), 0, -1):
        if s.endswith(marker[:n]):
            return n
    return 0
