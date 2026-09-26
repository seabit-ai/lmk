"""exp01: a request with tools, draft off vs on, greedy sha + drafted/accepted by segment.

usage (from the lmk checkout, engine at ENGINE_COMMIT):
  PYTHONPATH=.engine/mlx-engine:. .venv/bin/python research/2026-09-25-spec-with-tools/exp01-tools-draft-sha/run.py <out_dir>
env: KV_BITS (default 8), EFFORT (default low), THINKING (on|off, default on), TASK (fizzbuzz|copy, default fizzbuzz)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from lmk.config import load_config
from lmk.engine import MlxEngine
from lmk.models import resolve_draft, resolve_model
from lmk.sampling import parse_sampling

KV_BITS = int(os.environ.get("KV_BITS") or 8)
EFFORT = os.environ.get("EFFORT") or "low"
THINKING = (os.environ.get("THINKING") or "on") == "on"
MAX_TOKENS = 400

TOOLS = [
    {"type": "function", "function": {
        "name": "file_read", "description": "Read a text file and return its contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "file_write", "description": "Write a text file, replacing it if it exists.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the project directory and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
MESSAGES = [
    {"role": "system", "content": "You are kitten, a coding agent working in the user's project. Use the tools to act; "
                                  "keep prose short."},
    {"role": "user", "content": "Create fizzbuzz.py: a function fizzbuzz(n) returning the list of strings for 1..n "
                                "(Fizz for multiples of 3, Buzz for 5, FizzBuzz for both), a docstring, and a "
                                "__main__ block that prints fizzbuzz(30) one per line. Then run it."},
]

COPY = ("# Release checklist\n\n1. Run make test, make lint and make itest; all three must pass.\n"
        "2. Install the branch with make install and run lmk status; it must show answering 0 before you restart.\n"
        "3. Tag the release on main as vX.Y.Z and push the tag.\n4. Check that the install script picks up the new tag.\n")
if os.environ.get("TASK") == "copy":
    MESSAGES = [MESSAGES[0], {"role": "user", "content": "Save this text exactly as RELEASE.md, then show me the file "
                                                        "with cat:\n\n" + COPY}]


def segment(text: str) -> str:
    if THINKING and "</think>" not in text:
        return "thinking"
    after = text.rsplit("</think>", 1)[-1]
    opened, closed = after.rfind("<tool_call>"), after.rfind("</tool_call>")
    if opened > closed:
        return "body"
    if closed >= 0:
        return "tail"
    return "text"


def run(engine, prompt, name, *, spec, sampling):
    counters = engine._drafter_counters
    by_seg = {}
    text = ""
    t0 = time.perf_counter()
    first = None
    before = counters()
    last = before
    gen = engine.generate(prompt, max_tokens=MAX_TOKENS, request_id=name, on_prefill=lambda *a: True,
                          sampling={**sampling, "speculative_decoding_toggle": spec})
    for piece in gen.pieces:
        if first is None:
            first = time.perf_counter()
        text += piece
        now = counters()
        if now != last:
            seg = by_seg.setdefault(segment(text), {"rounds": 0, "accepted": 0, "drafted": 0})
            seg["rounds"] += now[0] - last[0]
            seg["accepted"] += now[1] - last[1]
            seg["drafted"] += now[2] - last[2]
            last = now
    end = time.perf_counter()
    n = gen.stats.completion_tokens
    total = {"rounds": last[0] - before[0], "accepted": last[1] - before[1], "drafted": last[2] - before[2]}
    return dict(name=name, spec=spec, tokens=n, sha=hashlib.sha1(text.encode()).hexdigest()[:12], text=text,
                decode_tok_s=round((n - 1) / max(1e-6, end - first), 1), total=total, by_segment=by_seg,
                ends_in=segment(text))


def show(r, ref=None):
    same = "" if ref is None else ("  identical" if r["sha"] == ref["sha"] else "  DIFFERENT")
    print(f"{r['name']:10} tokens {r['tokens']:3}  {r['decode_tok_s']:6.1f} tok/s  sha {r['sha']}{same}  ends in {r['ends_in']}",
          flush=True)
    if r["spec"]:
        t = r["total"]
        print(f"{'':10} drafted {t['drafted']}  accepted {t['accepted']}  rounds {t['rounds']}", flush=True)
        for seg, c in r["by_segment"].items():
            rate = c["accepted"] / c["drafted"] if c["drafted"] else 0
            print(f"{'':10}   {seg:8} drafted {c['drafted']:4}  accepted {c['accepted']:4}  ({rate:.0%})  rounds {c['rounds']}",
                  flush=True)


def first_difference(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    source = load_config().model.source
    model = resolve_model(source)
    draft = resolve_draft(source, None)
    print("task", os.environ.get("TASK") or "fizzbuzz", "\nmodel", model.path, "\ndraft", draft, "\nkv bits", KV_BITS, "thinking", THINKING, "effort", EFFORT, flush=True)
    kwargs = {"enable_thinking": THINKING}
    if THINKING and EFFORT != "none":
        kwargs["reasoning_effort"] = EFFORT
    engine = MlxEngine("exp01", model.path, 32768, template_kwargs=kwargs, kv_cache_bits=KV_BITS,
                       draft_path=draft, draft_kind="dflash2")
    prompt = engine.chat_format().render(MESSAGES, TOOLS)
    greedy = {"temp": 0.0}
    defaults, _ = parse_sampling({}, engine.sampling_defaults())
    defaults.pop("stop_strings", None)
    print("sampling defaults", defaults, flush=True)

    off1 = run(engine, prompt, "off-cold", spec=False, sampling=greedy); show(off1)
    off2 = run(engine, prompt, "off-warm", spec=False, sampling=greedy); show(off2, off1)
    on1 = run(engine, prompt, "on-1", spec=True, sampling=greedy); show(on1, off2)
    on2 = run(engine, prompt, "on-2", spec=True, sampling=greedy); show(on2, off2)
    for r in (on1, on2):
        if r["sha"] != off2["sha"]:
            i = first_difference(r["text"], off2["text"])
            print(f"{r['name']} first differs at char {i} (segment {segment(off2['text'][:i + 1])}): "
                  f"off {off2['text'][max(0, i - 40):i + 40]!r} / on {r['text'][max(0, i - 40):i + 40]!r}", flush=True)
    sampled = run(engine, prompt, "on-sampled", spec=True, sampling=defaults); show(sampled)
    (out / "results.json").write_text(json.dumps(dict(kv_bits=KV_BITS, thinking=THINKING, effort=EFFORT, prompt=prompt,
                                                      runs=[off1, off2, on1, on2, sampled]), indent=1, ensure_ascii=False))
    print("--- off-warm text ---\n" + off2["text"], flush=True)
    engine.close()


if __name__ == "__main__":
    main()
