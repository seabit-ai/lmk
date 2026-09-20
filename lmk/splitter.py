"""Streaming three-way split of raw model output: reasoning / answer text /
tool-call blocks. The engine emits plain text (research LMS-014); which markers
delimit a tool call is the model family's business and arrives from outside
(design §4) — this file knows only `</think>`.
"""

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_WS = " \t\r\n"


class OutputSplitter:
    def __init__(self, tool_call_start, tool_call_end, starts_in_reasoning, on_reasoning, on_text, on_tool_block):
        self._tool_start = tool_call_start
        self._tool_end = tool_call_end
        self._on_reasoning = on_reasoning
        self._on_text = on_text
        self._on_tool_block = on_tool_block
        # "leading": nothing decided yet (the model may or may not open a think block itself)
        self._phase = "reasoning" if starts_in_reasoning else "leading"
        self._pending = ""

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
            if trimmed.startswith(THINK_OPEN):
                self._pending, self._phase = trimmed[len(THINK_OPEN):], "reasoning"
                return True
            if THINK_OPEN.startswith(trimmed):
                return False
            self._phase = "text"
            return True
        if self._phase == "reasoning":
            return self._until(THINK_CLOSE, self._on_reasoning, "gap")
        if self._phase == "gap":
            self._pending = p.lstrip(_WS)
            if not self._pending:
                return False
            self._phase = "text"
            return True
        if self._phase == "text":
            if not self._tool_start:
                self._emit(self._on_text, p)
                self._pending = ""
                return False
            return self._until(self._tool_start, self._on_text, "tool")
        # "tool": hold everything until the end marker, then hand over the whole block
        i = p.find(self._tool_end)
        if i < 0:
            return False
        self._on_tool_block(p[:i])
        self._pending, self._phase = p[i + len(self._tool_end):], "gap"
        return True

    def _until(self, marker: str, emit, next_phase: str) -> bool:
        p = self._pending
        i = p.find(marker)
        if i >= 0:
            self._emit(emit, p[:i])
            self._pending, self._phase = p[i + len(marker):], next_phase
            return True
        keep = _partial_suffix_len(p, marker)
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
