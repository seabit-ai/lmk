"""The memory numbers lmk reports and acts on (design memory-guard §E): what macOS itself
says about the whole machine. Numbers only — lmk does not know who is using the memory,
so it never says."""
import ctypes
import ctypes.util
from dataclasses import dataclass

_PRESSURE = {1: "normal", 2: "warning", 4: "critical"}


@dataclass(frozen=True)
class MemoryReading:
    pressure: str       # "normal" | "warning" | "critical" — the line in Activity Monitor's memory graph
    free_percent: int   # macOS's own figure for the whole machine
    total_bytes: int

    @property
    def critical(self) -> bool:
        return self.pressure == "critical"


def _sysctl_int(name: str, size: int) -> int:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    buf = ctypes.c_uint64(0) if size == 8 else ctypes.c_uint32(0)
    length = ctypes.c_size_t(ctypes.sizeof(buf))
    if libc.sysctlbyname(name.encode(), ctypes.byref(buf), ctypes.byref(length), None, 0) != 0:
        raise OSError(ctypes.get_errno(), f"sysctl {name}")
    return int(buf.value)


class SystemMemory:
    def read(self) -> MemoryReading:
        level = _sysctl_int("kern.memorystatus_vm_pressure_level", 4)
        return MemoryReading(pressure=_PRESSURE.get(level, "normal"),
                             free_percent=_sysctl_int("kern.memorystatus_level", 4),
                             total_bytes=_sysctl_int("hw.memsize", 8))


_current = SystemMemory()


def get_current_memory():
    return _current


def set_current_memory(memory) -> None:
    global _current
    _current = memory
