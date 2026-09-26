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

    # Per bits x context, the worst a decode request made the process hold (the long prefix restored from
    # the cache, then 256 tokens): max(footprint polled every 0.5 s, MLX peak + the non-MLX part). The peak
    # is instantaneous (restoring the prefix) and the poll can miss it; the non-MLX part (Metal/driver/
    # Python, footprint − active − cache) is the median over exp01. Cold prefill is left out on purpose: its
    # peak depends on the prefill step, which the engine lowers on a smaller Mac (its formula covers that);
    # restoring a cached prefix is not in the formula.
    extra = statistics.median((r["mem"]["footprint_max"] or 0) - (r["mem"]["held_max"] or 0) for r in e1)
    worst = defaultdict(dict)   # (model, bits) -> {tokens: bytes}

    def add(model, bits, r):
        if r["mode"] != "greedy":
            return
        tok = r["done"]["promptTokens"] + r["done"]["completionTokens"]
        v = max(r["mem"]["footprint_max"] or 0, (r["mem"]["peak"] or 0) + extra)
        d = worst[(model, bits)]
        d[tok] = max(d.get(tok, 0), v)

    for r in e1:
        add("uncapped", int(r["cond"][2:]), r)
    for bits, r in capped_runs:
        if r["cond"] == "limit1g":
            add("capped", bits, r)

    lines = {}
    print("## Fitted lines (96 GB Mac, one request at a time; t = tokens in the request)\n")
    print(f"worst held = max(footprint, MLX peak + {extra / GB:.2f} GB non-MLX). Uncapped: exp01 decode requests at "
          "8k/32k/64k/128k. Capped (1 GiB): exp02 (kv16) / exp04 (kv8) at 32k/128k.\n")
    print("| model | bits | points (tokens: GB) | worst held (GB) = a + b·t | b ÷ KV B/token |")
    print("|---|---|---|---|---|")
    for (model, bits), d in sorted(worst.items()):
        a, b = fit_line(sorted(d.items()))
        lines[(model, bits)] = (a, b)
        pts = ", ".join(f"{t // 1000}k: {v / GB:.1f}" for t, v in sorted(d.items()))
        print(f"| {model} | kv{bits} | {pts} | {a / GB:.1f} + {b / 1e3:.0f} kB × t | {b / KV[bits]:.1f}× |")

    print(f"\n## Per Mac size (computed; only 96 GB was measured). Capped: MLX cache limit {limit / GIB:.0f} GiB\n")
    print("working set = GB × 0.81 GiB (lmk's GPU_SHARE_OF_MEMORY). Window and cap: lmk's formulas (`context_on`, "
          "`engine.token_budget`). 'At full window': the lines above evaluated at the formula window (beyond 131k "
          "extrapolated), minus the working set. 'Largest context': where the line meets the working set — also what "
          "one request could really use of the tokens-in-memory cap.\n")
    print("| Mac | bits | window (formula) | tokens-in-memory cap (formula) | at full window vs working set, uncapped (GB) | "
          "capped | largest context under the working set, uncapped | capped |")
    print("|---|---|---|---|---|---|---|---|")
    k = lambda n: f"{n / 1000:.0f}k" if n < 10 ** 6 else f"{n / 1e6:.2f}M"  # noqa: E731
    for gb in SIZES:
        ws = gb * GPU_SHARE_OF_MEMORY * GIB
        for bits in (8, 16):
            win = context_on(M, gb, bits)
            avail = ws - ENGINE_RESERVE_BYTES - F.baseline_gib * GIB
            cap = int(max(0, avail) // KV[bits])
            if not win or avail <= 0:
                print(f"| {gb} GB | kv{bits} | does not load | – | – | – | – | – |")
                continue
            cells = []
            for model in ("uncapped", "capped"):
                a, b = lines[(model, bits)]
                cells.append(((a + b * win - ws) / GB, min(M.max_context, max(0, int((ws - a) // b)))))
            print(f"| {gb} GB | kv{bits} | {k(win)} | {k(cap)} | {cells[0][0]:+.1f} | {cells[1][0]:+.1f} | "
                  f"{k(cells[0][1])} | {k(cells[1][1])} |")

if __name__ == "__main__":
    main()
