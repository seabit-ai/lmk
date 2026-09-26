"""results.md: medians per condition x context (exp01 and exp02 record the same fields; the file is copied).

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
    print("## Engine fit and lmk's token cap (the temporary server's own log / status)\n")
    print("| condition | KV B/token (engine) | context window | token_budget (\"tokens in memory\" cap) | working set | baseline |")
    print("|---|---|---|---|---|---|")
    for c in conds:
        f = raw / f"context-fit-{c}.txt"
        m = re.search(r"working_set=([\d.]+)GiB .*baseline=([\d.]+)GiB full_kv=(\d+)B/token", f.read_text() if f.exists() else "")
        st = json.loads((raw / f"status-{c}-start.json").read_text())
        kv[c] = int(m.group(3)) if m else None
        print(f"| {c} | {kv[c] or 0:,} | {st['model']['context_length']:,} | {st['requests']['token_budget'] or 0:,} | "
              f"{m.group(1) if m else '?'} GiB | {m.group(2) if m else '?'} GiB |")

    by = defaultdict(list)
    for r in runs:
        by[(r["cond"], r["ctx"], r["mode"])].append(r)
    ctxs = sorted({r["ctx"] for r in runs})

    print("\n## Decode requests (greedy, 256 tokens, prefix cached): medians over kinds x reps, GB (10^9)\n")
    print("During = max over the request (0.25–0.5 s polls); after = last poll 5 s after the response. "
          "Peak is per request (reset at its start).\n")
    print("| condition | context | KV bytes | footprint during | MLX active | MLX cache | active+cache | MLX peak | "
          "footprint after | active / cache after | ps RSS during / after | decode tok/s, each request | ttft ms | n |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        for ctx in ctxs:
            rs = by.get((c, ctx, "greedy"), [])
            if not rs:
                continue
            m = lambda k: med([r["mem"].get(k) for r in rs])  # noqa: E731
            toks = med([r["done"]["promptTokens"] + r["done"]["completionTokens"] for r in rs])
            speeds = " / ".join(f"{r['kind'][0]} {r['decode_tps']:.1f}" for r in rs)
            print(f"| {c} | {ctx // 1024}k | {g(kv[c] * toks if kv[c] else None)} | {g(m('footprint_max'))} | {g(m('active_max'))} | "
                  f"{g(m('cache_max'))} | {g(m('held_max'))} | {g(m('peak'))} | {g(m('footprint_after'))} | "
                  f"{g(m('active_after'))} / {g(m('cache_after'))} | {g(m('rss_max'))} / {g(m('rss_after'))} | "
                  f"{speeds} | {med([r['done']['ttftMs'] for r in rs]):,.0f} | {len(rs)} |")

    print("\n## Warm requests (max_tokens 1): cached / prompt tokens shows cold prefill vs restore from the cache; GB\n")
    print("| condition | context | kind | cached / prompt tokens | ttft ms | footprint during | active+cache | MLX peak | footprint after | active / cache after |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        for ctx in ctxs:
            for r in by.get((c, ctx, "warm"), []):
                d, mm = r["done"], r["mem"]
                print(f"| {c} | {ctx // 1024}k | {r['kind']} | {d['cachedTokens']:,} / {d['promptTokens']:,} | {d['ttftMs']:,} | "
                      f"{g(mm.get('footprint_max'))} | {g(mm['held_max'])} | {g(mm['peak'])} | {g(mm.get('footprint_after'))} | "
                      f"{g(mm['active_after'])} / {g(mm['cache_after'])} |")


if __name__ == "__main__":
    main()
