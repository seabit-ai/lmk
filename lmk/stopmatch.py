"""OpenAI `stop` on the answer part only.

The engine's own stop_strings watch the whole token stream, thinking included,
and on a thinking model that empties the answer (research/2026-09-21-sampling SMP-002).
So lmk keeps `stop` to itself: this sits behind the output splitter, sees only
answer text, and tells run_chat to close the engine's generator on a match."""
from typing import Callable


class StopMatcher:
    def __init__(self, stops: list[str], on_text: Callable[[str], None]):
        self._stops = [s for s in stops if s]
        self._on_text = on_text
        self._held = ""   # tail that might be the start of a stop string
        self._hold = max((len(s) for s in self._stops), default=1) - 1

    def write(self, piece: str) -> bool:
        """Emits what is safe to emit. True = a stop string was hit; the text before it is out, nothing after."""
        if not self._stops:
            self._on_text(piece)
            return False
        text = self._held + piece
        hits = [(text.find(s), s) for s in self._stops if s in text]
        if hits:
            at, _ = min(hits)
            if at:
                self._on_text(text[:at])
            self._held = ""
            return True
        keep = 0
        for n in range(min(self._hold, len(text)), 0, -1):   # longest tail that prefixes some stop string
            if any(s.startswith(text[-n:]) for s in self._stops):
                keep = n
                break
        if len(text) > keep:
            self._on_text(text[:len(text) - keep])
        self._held = text[len(text) - keep:] if keep else ""
        return False

    def close(self) -> None:
        """Generation ended on its own: a held tail was not a stop string after all."""
        if self._held:
            self._on_text(self._held)
            self._held = ""
