"""exp03 requests: 32k prefix, prose and code, greedy and the model's default sampling, in two profile modes.

usage: measure.py <base_url> <lmk_log_jsonl> <prefix_file> <control_file> <out_jsonl>
Per kind: one warm request (max_tokens 1), then for mode in (wall, phases), for sampling in (greedy, sampled):
REPS requests of max_tokens 256. The control file tells profile_patch the mode and a tag for the request.
The resident lmk is polled as in exp01 (it was stopped for this run; the polls record that).
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp01-context-sweep"))
from measure import MODEL, TASKS, ResidentWatch, done_line, post  # noqa: E402  (exp01's harness, reused unchanged)

REPS = 3
MAX_TOKENS = 256


def main():
    base, log_path, prefix_file, control, out = sys.argv[1:6]
    prefix = Path(prefix_file).read_text(encoding="utf-8")
    with open(out, "a") as f:
        for kind, task in TASKS.items():
            msgs = [{"role": "user", "content": prefix + "\n\n" + task}]
            Path(control).write_text(f"wall {kind}/warm")
            ref = f"{kind}/warm"
            post(base, {"model": MODEL, "messages": msgs, "max_tokens": 1, "temperature": 0}, ref)
            w = done_line(log_path, ref)
            print(f"{kind} warm: prompt {w['promptTokens']} cached {w['cachedTokens']} ttft {w['ttftMs']} ms", flush=True)
            f.write(json.dumps({"kind": kind, "mode": "warm", "done": w}) + "\n")
            for prof in ("wall", "phases"):
                for sampling, extra in (("greedy", {"temperature": 0}), ("sampled", {})):
                    for rep in range(1, REPS + 1):
                        for attempt in range(1, 7):
                            tag = f"{kind}/{prof}/{sampling}/{rep}/a{attempt}"
                            Path(control).write_text(f"{prof} {tag}")
                            with ResidentWatch() as watch:
                                post(base, {"model": MODEL, "messages": msgs, "max_tokens": MAX_TOKENS, **extra}, tag)
                            d = done_line(log_path, tag)
                            if not watch.busy:
                                break
                            f.write(json.dumps({"kind": kind, "mode": "discarded", "tag": tag, "done": d}) + "\n")
                            print(f"{tag}: DISCARDED, resident busy", flush=True)
                            time.sleep(20)
                        else:
                            raise RuntimeError("resident busy on every attempt")
                        dec_ms = d["totalMs"] - d["ttftMs"]
                        tps = (d["completionTokens"] - 1) / dec_ms * 1000
                        f.write(json.dumps({"kind": kind, "mode": "measured", "prof": prof, "sampling": sampling, "rep": rep,
                                            "tag": tag, "decode_tps": round(tps, 2), "resident_poll_errors": watch.errors,
                                            "done": d}) + "\n")
                        f.flush()
                        print(f"{tag}: decode {tps:.1f} tok/s drafted {d.get('draftDrafted')} accepted {d.get('draftAccepted')}",
                              flush=True)
    Path(control).write_text("wall none")


if __name__ == "__main__":
    main()
