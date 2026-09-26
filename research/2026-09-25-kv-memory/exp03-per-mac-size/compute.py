"""exp03: per Mac size, kv8 vs kv16 — the engine's window and lmk's tokens-in-memory cap (lmk's formulas),
and what a process would really hold with the window full (formula + what exp01/exp02 measured on the 96 GB Mac).

usage: compute.py <exp01 runs.jsonl> <exp02 runs.jsonl (kv16)> <exp04 runs.jsonl (kv8)> <cache limit bytes for the capped column>
Run with the lmk checkout on the import path (PYTHONPATH=<worktree>).
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from lmk.models import (ENGINE_RESERVE_BYTES, GIB, GPU_SHARE_OF_MEMORY, MAC_MEMORY_SIZES_GB, TESTED_MODELS,
                        context_on)

M = TESTED_MODELS["qwen3.8-27b-4bit"]
F = M.fit
KV = {8: F.full_kv_bytes_per_token_8bit, 16: F.full_kv_bytes_per_token}
SIZES = [g for g in MAC_MEMORY_SIZES_GB if g <= 192]
GB = 1e9


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def fit_line(pts):
    """least squares y = a + b x"""
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    return my - b * mx, b


def main():
    e1, limit = load(sys.argv[1]), int(sys.argv[4])
    capped_runs = [(16, r) for r in load(sys.argv[2])] + [(8, r) for r in load(sys.argv[3])]

    # exp01: per bits x context, the worst the process held over the decode requests at that context
    # (the long prefix restored from the cache, then 256 tokens) and their MLX peak. Cold prefill is left
    # out on purpose: its peak depends on the prefill step, which the engine lowers on a smaller Mac
    # (its formula covers that case); restoring a cached prefix is not in the formula.
    worst = defaultdict(lambda: {"fp": 0, "peak": 0, "held": 0})
    for r in e1:
        if r["mode"] != "greedy":
            continue
        bits = int(r["cond"][2:])
        tok = r["done"]["promptTokens"] + r["done"]["completionTokens"]
        w = worst[(bits, r["ctx"])]
        w["tok"] = max(w.get("tok", 0), tok)
        w["fp"] = max(w["fp"], r["mem"]["footprint_max"] or 0)
        w["peak"] = max(w["peak"], r["mem"]["peak"] or 0)
        w["held"] = max(w["held"], r["mem"]["held_max"] or 0)
    # footprint beyond MLX's own active + cache (Metal/driver/Python): median over all exp01 requests
    extra = statistics.median((r["mem"]["footprint_max"] or 0) - (r["mem"]["held_max"] or 0) for r in e1)

    lines = {}
    print("## Fitted lines from exp01 decode requests (96 GB Mac, one request at a time, 8k–128k; x = tokens in the request)\n")
    print("| bits | footprint during (GB) = a + b·tokens | b ÷ KV B/token | MLX peak (GB) = a + b·tokens | b ÷ KV B/token |")
    print("|---|---|---|---|---|")
    for bits in (8, 16):
        pts = sorted((w["tok"], w) for (b, _), w in worst.items() if b == bits)
        fa, fb = fit_line([(t, w["fp"]) for t, w in pts])
        pa, pb = fit_line([(t, w["peak"]) for t, w in pts])
        lines[bits] = {"fp": (fa, fb), "peak": (pa, pb)}
        print(f"| {bits} | {fa / GB:.1f} + {fb / 1e3:.0f} kB × t | {fb / KV[bits]:.1f}× | {pa / GB:.1f} + {pb / 1e3:.0f} kB × t | {pb / KV[bits]:.1f}× |")
    print(f"\nfootprint − (MLX active + cache), median over exp01 requests: {extra / GB:.2f} GB")

    # the capped model: footprint <= peak + limit + extra. Check it against exp02 (kv16, capped conditions).
    print(f"\n## Check of the capped model on exp02 (kv16) and exp04 (kv8): footprint during ≤ MLX peak + limit + {extra / GB:.2f} GB\n")
    print("| bits | condition | context | limit | footprint during (measured, max of reps) | peak + limit + extra | holds |")
    print("|---|---|---|---|---|---|---|")
    lim_of = {}
    for bits, r in capped_runs:
        if r["mode"] != "greedy":
            continue
        c = r["cond"]
        lim = {"limit0": 0, "limit1g": 1 << 30, "limit4g": 4 << 30}.get(c)
        if lim is None:
            continue
        k = (bits, c, r["ctx"])
        lim_of.setdefault(k, [lim, 0, 0])
        lim_of[k][1] = max(lim_of[k][1], r["mem"]["footprint_max"] or 0)
        lim_of[k][2] = max(lim_of[k][2], r["mem"]["peak"] or 0)
    for (bits, c, ctx), (lim, fp, peak) in sorted(lim_of.items()):
        bound = peak + lim + extra
        print(f"| kv{bits} | {c} | {ctx // 1024}k | {lim / GIB:.0f} GiB | {fp / GB:.1f} | {bound / GB:.1f} | {'yes' if fp <= bound * 1.02 else 'NO'} |")

    print(f"\n## Per Mac size (computed; only 96 GB was measured). Capped column: MLX cache limit {limit / GIB:.0f} GiB\n")
    print("working set = GB × 0.81 GiB (lmk's GPU_SHARE_OF_MEMORY). Window and cap: lmk's formulas (`context_on`, "
          "`engine.token_budget`). 'Holds at full window': the exp01 lines evaluated at the window (beyond 131k they are "
          "extrapolated), uncapped = footprint line, capped = peak line + limit + extra.\n")
    print("| Mac | bits | window (formula) | tokens-in-memory cap (formula) | holds at full window, uncapped: GB (over/under working set) | "
          "capped | largest context under the working set, uncapped | capped |")
    print("|---|---|---|---|---|---|---|---|")
    for gb in SIZES:
        ws = gb * GPU_SHARE_OF_MEMORY * GIB
        for bits in (8, 16):
            win = context_on(M, gb, bits)
            avail = ws - ENGINE_RESERVE_BYTES - F.baseline_gib * GIB
            cap = int(max(0, avail) // KV[bits])
            if not win or avail <= 0:
                print(f"| {gb} GB | kv{bits} | does not load | – | – | – | – | – |")
                continue
            fa, fb = lines[bits]["fp"]
            pa, pb = lines[bits]["peak"]
            un = fa + fb * win
            ca = pa + pb * win + limit + extra
            safe_un = max(0, int((ws - fa) // fb))
            safe_ca = max(0, int((ws - pa - limit - extra) // pb))
            k = lambda n: f"{n / 1000:.0f}k" if n < 10 ** 6 else f"{n / 1e6:.2f}M"  # noqa: E731
            print(f"| {gb} GB | kv{bits} | {k(win)} | {k(cap)} | {un / GB:.1f} ({(un - ws) / GB:+.1f}) | {ca / GB:.1f} ({(ca - ws) / GB:+.1f}) | "
                  f"{k(min(safe_un, M.max_context))} | {k(min(safe_ca, M.max_context))} |")


if __name__ == "__main__":
    main()
