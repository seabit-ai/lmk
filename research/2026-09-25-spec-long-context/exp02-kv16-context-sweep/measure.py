"""Send the measured requests to one running lmk and record each one's LmkChatDone line.
exp02 copy of exp01's measure.py, plus two readings of the temporary lmk itself: its GPU memory
(max of lmk_gpu_bytes polled every 0.5 s while the request runs, and the process peak afterwards)
and its speculative round count before/after the request (exact rounds, exp01 had to estimate them).

usage: measure.py <base_url> <lmk_log_jsonl> <prefix_dir> <condition> <out_jsonl> <ctx> [<ctx> ...]
For each context: for each kind (prose, code): one warm request (identical prompt, max_tokens 1),
then REPS greedy requests of max_tokens 256. At SAMPLED_CTX also REPS requests per kind with the
model's default sampling (no temperature in the request).
"""
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

MODEL = "qwen3.8-27b-4bit"
REPS = int(os.environ.get("REPS") or 3)
MAX_TOKENS = 256
SAMPLED_CTX = int(os.environ.get("SAMPLED_CTX") or 32768)
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


RESIDENT = os.environ.get("RESIDENT") or "http://127.0.0.1:1235"
MAX_ATTEMPTS = 6


class ResidentWatch:
    """Polls the resident lmk every 0.5 s while a measured request runs; busy = it had a request in flight
    or its answered count moved (a request that started and ended between two polls)."""

    def __init__(self):
        self.busy, self.seen, self.errors, self._stop = False, [], 0, threading.Event()

    def _read(self):
        with urllib.request.urlopen(RESIDENT + "/lmk/v1/status", timeout=5) as r:
            s = json.load(r)
        return s["totals"]["answered"], len(s.get("in_flight") or []), s["requests"]["answering"] + s["requests"]["waiting"]

    def _loop(self):
        first = None
        while not self._stop.is_set():
            try:
                answered, flight, active = self._read()
            except Exception:
                self.errors += 1
            else:
                first = answered if first is None else first
                if flight or active or answered != first:
                    self.busy = True
                    self.seen.append({"answered": answered, "in_flight": flight, "active": active})
            self._stop.wait(0.5)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        time.sleep(0.6)  # one poll lands before the request starts
        return self

    def __exit__(self, *exc):
        time.sleep(0.6)
        self._stop.set()
        self._t.join()


class SelfWatch:
    """Polls the temporary lmk's own status while a measured request runs."""

    def __init__(self, base):
        self.base, self.max_gpu, self._stop = base, 0, threading.Event()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.max_gpu = max(self.max_gpu, status(self.base)["memory"]["lmk_gpu_bytes"])
            except Exception:
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def status(base):
    with urllib.request.urlopen(base + "/lmk/v1/status", timeout=5) as r:
        return json.load(r)


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


def main():
    base, log_path, pdir, cond, out = sys.argv[1:6]
    ctxs = [int(c) for c in sys.argv[6:]]
    with open(out, "a") as f:
        for ctx in ctxs:
            prefix = (Path(pdir) / f"prefix-{ctx}.txt").read_text(encoding="utf-8")
            runs = [("greedy", {"temperature": 0})]
            if ctx == SAMPLED_CTX:
                runs.append(("sampled", {}))
            for kind, task in TASKS.items():
                msgs = [{"role": "user", "content": prefix + "\n\n" + task}]
                ref = f"{cond}/{ctx}/{kind}/warm"
                t0 = time.time()
                post(base, {"model": MODEL, "messages": msgs, "max_tokens": 1, "temperature": 0}, ref)
                w = done_line(log_path, ref)
                print(f"{cond} {ctx} {kind} warm: prompt {w['promptTokens']} cached {w['cachedTokens']} "
                      f"ttft {w['ttftMs']} ms ({time.time() - t0:.0f}s wall)", flush=True)
                f.write(json.dumps({"cond": cond, "ctx": ctx, "kind": kind, "mode": "warm", "rep": 0, "done": w}) + "\n")
                for mode, extra in runs:
                    for rep in range(1, REPS + 1):
                        for attempt in range(1, MAX_ATTEMPTS + 1):
                            ref = f"{cond}/{ctx}/{kind}/{mode}/{rep}/a{attempt}"
                            before = status(base)
                            with ResidentWatch() as watch, SelfWatch(base) as mem:
                                resp = post(base, {"model": MODEL, "messages": msgs, "max_tokens": MAX_TOKENS, **extra}, ref)
                            after = status(base)
                            d = done_line(log_path, ref)
                            if not watch.busy:
                                break
                            f.write(json.dumps({"cond": cond, "ctx": ctx, "kind": kind, "mode": "discarded", "rep": rep,
                                                "attempt": attempt, "resident": watch.seen[:5], "done": d}) + "\n")
                            print(f"{cond} {ctx} {kind} {mode} #{rep}: DISCARDED, resident busy {watch.seen[:2]}", flush=True)
                            time.sleep(20)
                        else:
                            raise RuntimeError("resident busy on every attempt")
                        text = resp["choices"][0]["message"].get("content") or ""
                        dec_ms = d["totalMs"] - d["ttftMs"]
                        tps = (d["completionTokens"] - 1) / dec_ms * 1000 if dec_ms > 0 else 0.0
                        rec = {"cond": cond, "ctx": ctx, "kind": kind, "mode": mode, "rep": rep, "done": d,
                               "decode_tps": round(tps, 2), "attempt": attempt,
                               "rounds": (after.get("draft") or {}).get("rounds", 0) - (before.get("draft") or {}).get("rounds", 0),
                               "gpu_bytes_max": mem.max_gpu, "gpu_peak_in_use_bytes": after["memory"]["lmk_gpu_peak_in_use_bytes"], "resident_poll_errors": watch.errors, "sha": hashlib.sha256(text.encode()).hexdigest()[:12],
                               "text": text}
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        print(f"{cond} {ctx} {kind} {mode} #{rep}: prompt {d['promptTokens']} cached {d['cachedTokens']} "
                              f"completion {d['completionTokens']} {d['finishReason']} ttft {d['ttftMs']} ms "
                              f"decode {tps:.1f} tok/s drafted {d.get('draftDrafted')} accepted {d.get('draftAccepted')} "
                              f"sha {rec['sha']} rounds {rec['rounds']} gpu {mem.max_gpu / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
