"""results.md for exp01 (and exp02, which records the same fields): medians per condition x context.

usage: table.py <runs.jsonl> [<raw dir>]   (raw dir: where status-<cond>-start.json and context-fit-<cond>.txt live)
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GB = 1e9


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def g(b):
    return f"{b / GB:.1f}" if b else "–"


def main():
    runs = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    raw = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(sys.argv[1]).parent
    conds = list(dict.fromkeys(r["cond"] for r in runs))
    kv = {}
    print("## Engine fit and lmk's token cap (from the temporary server's own log / status)\n")
    print("| condition | KV B/token (engine) | context window | token_budget (\"tokens in memory\" cap) | working set | baseline |")
    print("|---|---|---|---|---|---|")
    for c in conds:
        fit = (raw / f"context-fit-{c}.txt").read_text() if (raw / f"context-fit-{c}.txt").exists() else ""
        m = re.search(r"working_set=([\d.]+)GiB .*baseline=([\d.]+)GiB full_kv=(\d+)B/token", fit)
        st = json.loads((raw / f"status-{c}-start.json").read_text())
        kv[c] = int(m.group(3)) if m else None
        print(f"| {c} | {kv[c]:,} | {st['model']['context_length']:,} | {st['requests']['token_budget']:,} | "
              f"{m.group(1) if m else '?'} GiB | {m.group(2) if m else '?'} GiB |")

    by = defaultdict(list)
    for r in runs:
        by[(r["cond"], r["ctx"], r["mode"])].append(r)
    ctxs = sorted({r["ctx"] for r in runs})

    print("\n## Decode requests (greedy, 256 tokens, prefix cached): medians over all kinds x reps, GB\n")
    print("| condition | context | KV bytes (engine B/token x tokens) | ps resident max | MLX active max | MLX cache max | "
          "active+cache max | MLX peak (this request) | ps resident 5 s after | active / cache 5 s after | decode tok/s code / prose | n |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        for ctx in ctxs:
            rs = by.get((c, ctx, "greedy"), [])
            if not rs:
                continue
            m = lambda k: med([r["mem"][k] for r in rs])  # noqa: E731
            toks = med([r["done"]["promptTokens"] + r["done"]["completionTokens"] for r in rs])
            tps = {k: med([r["decode_tps"] for r in rs if r["kind"] == k]) for k in ("code", "prose")}
            print(f"| {c} | {ctx // 1024}k | {g(kv[c] * toks if kv[c] else None)} | {g(m('rss_max'))} | {g(m('active_max'))} | "
                  f"{g(m('cache_max'))} | {g(m('held_max'))} | {g(m('peak'))} | {g(m('rss_after'))} | "
                  f"{g(m('active_after'))} / {g(m('cache_after'))} | "
                  f"{tps['code'] or 0:.1f} / {tps['prose'] or 0:.1f} | {len(rs)} |")

    print("\n## Warm requests (max_tokens 1): the first kind at each context prefills the new part of the prefix cold, "
          "the second restores it from the cache; GB\n")
    print("| condition | context | kind | cached / prompt tokens | ttft ms | ps resident max | active+cache max | MLX peak | ps resident 5 s after |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        for ctx in ctxs:
            for r in by.get((c, ctx, "warm"), []):
                d, mm = r["done"], r["mem"]
                print(f"| {c} | {ctx // 1024}k | {r['kind']} | {d['cachedTokens']:,} / {d['promptTokens']:,} | {d['ttftMs']:,} | "
                      f"{g(mm['rss_max'])} | {g(mm['held_max'])} | {g(mm['peak'])} | {g(mm['rss_after'])} |")


if __name__ == "__main__":
    main()
