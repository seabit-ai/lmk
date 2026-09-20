"""Injectable time, so nothing in lmk reads the wall clock directly (tests swap it)."""
import time


class SystemClock:
    def wall_ms(self) -> int:
        return int(time.time() * 1000)

    def mono_ms(self) -> int:
        return int(time.monotonic() * 1000)

    def sleep_s(self, seconds: float) -> None:
        time.sleep(seconds)


_current = SystemClock()


def get_current_clock():
    return _current


def set_current_clock(clock) -> None:
    global _current
    _current = clock
