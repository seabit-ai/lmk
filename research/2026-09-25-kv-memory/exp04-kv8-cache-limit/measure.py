"""Send the measured requests to one running lmk and record, per request, its LmkChatDone line and the
memory the server process holds: resident bytes from `ps` (lmk.memory.SystemMemory.resident_bytes, the
reading lmk itself uses) polled every 0.5 s during the request and for AFTER_S seconds after it, and the
MLX counters the probe (probe/sitecustomize.py) writes from inside the process.

usage: measure.py <base_url> <lmk_log_jsonl> <probe_jsonl> <server_pid> <prefix_dir> <condition> <out_jsonl> <ctx> [<ctx> ...]
For each context and kind: one warm request (max_tokens 1; the first one at a context prefills the
prefix cold, which is itself a memory reading), then REPS greedy requests of max_tokens 256.
The resident service was stopped by the owner before these runs, so there is no resident polling.
"""
import ctypes
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

from lmk.memory import SystemMemory

MODEL = "qwen3.8-27b-4bit"
REPS = int(os.environ.get("REPS") or 3)
KINDS = (os.environ.get("KINDS") or "code prose").split()
MAX_TOKENS = 256
AFTER_S = float(os.environ.get("AFTER_S") or 5)
TASKS = {
    "prose": "The text above is the source code of a Python package. Without writing any code, explain in plain "
             "English prose how this software is organized and how a request flows through it, module by module. "
             "Write at least 800 words in flowing paragraphs: no lists, no headings, no code blocks.",
    "code": "Write a complete, self-contained Python module that implements a thread-safe LRU cache class with get, "
            "put, delete, a size limit in bytes, per-entry expiry and eviction callbacks, followed by a full "
            "unittest test suite with at least fifteen test methods. Output only the code, no explanations.",
}


def post(base, body, ref):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "X-Lmk-Purpose": "exp",
                                          "X-Lmk-Ref-Id": ref})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def status(base):
    with urllib.request.urlopen(base + "/lmk/v1/status", timeout=5) as r:
        return json.load(r)


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def phys_footprint(pid):
    """The kernel's phys_footprint of a process (proc_pid_rusage, RUSAGE_INFO_V2): what Activity Monitor's
    Memory column shows and what memory pressure counts. ps's RSS can miss GPU (Metal) allocations."""
    buf = (ctypes.c_uint64 * 32)()
    if _libproc.proc_pid_rusage(pid, 2, ctypes.byref(buf)) != 0:
        return None
    return int(buf[2 + 7])  # after the 16-byte uuid: user, system, idle wkups, intr wkups, pageins, wired, resident, footprint


class RssWatch:
    """The server's resident bytes as ps reports them (lmk.memory, what lmk itself reads) and its
    phys_footprint, every 0.5 s: (wall ms, rss, footprint)."""

    def __init__(self, pid):
        self.pid, self.samples, self._stop = pid, [], threading.Event()

    def _loop(self):
        mem = SystemMemory()
        while not self._stop.is_set():
            b = mem.resident_bytes(self.pid)
            if b:
                self.samples.append((int(time.time() * 1000), b, phys_footprint(self.pid)))
            self._stop.wait(0.5)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def done_line(log_path, ref):
    for _ in range(50):
        for line in reversed(Path(log_path).read_text().splitlines()):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("event") == "LmkChatDone" and r.get("refId") == ref:
                return r
        time.sleep(0.2)
    raise RuntimeError(f"no LmkChatDone for {ref}")


def probe_lines(path, t0, t1):
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if t0 <= r.get("t_ms", 0) <= t1:
            out.append(r)
    return out


def summarize(probe, rss, t_start, t_end, t_after):
    during = [r for r in probe if r["ev"] == "S" and t_start <= r["t_ms"] <= t_end]
    after = [r for r in probe if r["ev"] == "S" and t_end < r["t_ms"] <= t_after]
    rss_during = [b for t, b, _ in rss if t_start <= t <= t_end]
    rss_after = [b for t, b, _ in rss if t_end < t <= t_after]
    fp_during = [f for t, _, f in rss if t_start <= t <= t_end and f]
    fp_after = [f for t, _, f in rss if t_end < t <= t_after and f]
    end = next((r for r in probe if r["ev"] == "GenEnd"), None)
    cleared = next((r for r in probe if r["ev"] == "Cleared"), None)
    return {
        "rss_max": max(rss_during, default=None), "rss_after": rss_after[-1] if rss_after else None,
        "footprint_max": max(fp_during, default=None), "footprint_after": fp_after[-1] if fp_after else None,
        "active_max": max((r["active"] for r in during), default=None),
        "cache_max": max((r["cache"] for r in during), default=None),
        "held_max": max((r["active"] + r["cache"] for r in during), default=None),  # what lmk status calls lmk_gpu_bytes
        "peak": end["peak"] if end else None,  # reset at the request's start by the probe
        "at_end": end, "after_clear": cleared,
        "active_after": after[-1]["active"] if after else None, "cache_after": after[-1]["cache"] if after else None,
        "n_probe": len(during), "n_rss": len(rss_during),
    }


def one(base, log_path, probe_path, pid, cond, ctx, kind, mode, rep, body, f):
    ref = f"{cond}/{ctx}/{kind}/{mode}/{rep}"
    before = status(base)
    with RssWatch(pid) as rss:
        time.sleep(0.6)
        t_start = int(time.time() * 1000)
        resp = post(base, {**body, "model": MODEL}, ref)
        t_end = int(time.time() * 1000)
        time.sleep(AFTER_S)
        t_after = int(time.time() * 1000)
    after = status(base)
    d = done_line(log_path, ref)
    mem = summarize(probe_lines(probe_path, t_start - 1000, t_after), rss.samples, t_start, t_end, t_after)
    text = resp["choices"][0]["message"].get("content") or ""
    dec_ms = d["totalMs"] - d["ttftMs"]
    tps = (d["completionTokens"] - 1) / dec_ms * 1000 if dec_ms > 0 and d["completionTokens"] > 1 else None
    rec = {"cond": cond, "ctx": ctx, "kind": kind, "mode": mode, "rep": rep, "done": d, "decode_tps": tps and round(tps, 2),
           "rounds": (after.get("draft") or {}).get("rounds", 0) - (before.get("draft") or {}).get("rounds", 0),
           "mem": mem, "status_memory": after["memory"], "requests": after["requests"],
           "sha": hashlib.sha256(text.encode()).hexdigest()[:12], "text": text}
    f.write(json.dumps(rec) + "\n")
    f.flush()
    g = lambda b: f"{b / 1e9:.2f}" if b else "-"  # noqa: E731
    print(f"{ref}: prompt {d['promptTokens']} cached {d['cachedTokens']} ttft {d['ttftMs']} ms "
          f"decode {tps or 0:.1f} tok/s | rss max {g(mem['rss_max'])} after {g(mem['rss_after'])}; "
          f"footprint max {g(mem['footprint_max'])} after {g(mem['footprint_after'])} GB; "
          f"active max {g(mem['active_max'])} cache max {g(mem['cache_max'])} peak {g(mem['peak'])} "
          f"after: active {g(mem['active_after'])} cache {g(mem['cache_after'])} GB", flush=True)


def main():
    base, log_path, probe_path, pid, pdir, cond, out = sys.argv[1:8]
    pid = int(pid)
    ctxs = [int(c) for c in sys.argv[8:]]
    with open(out, "a") as f:
        for ctx in ctxs:
            prefix = (Path(pdir) / f"prefix-{ctx}.txt").read_text(encoding="utf-8")
            for kind in KINDS:
                msgs = [{"role": "user", "content": prefix + "\n\n" + TASKS[kind]}]
                one(base, log_path, probe_path, pid, cond, ctx, kind, "warm", 0,
                    {"messages": msgs, "max_tokens": 1, "temperature": 0}, f)
                for rep in range(1, REPS + 1):
                    one(base, log_path, probe_path, pid, cond, ctx, kind, "greedy", rep,
                        {"messages": msgs, "max_tokens": MAX_TOKENS, "temperature": 0}, f)


if __name__ == "__main__":
    main()
