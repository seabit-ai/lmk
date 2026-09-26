"""Tables from raw/rounds.jsonl (one line per speculative round) and raw/runs.jsonl (one line per request).

usage: analyze.py <rounds.jsonl> <runs.jsonl>
All round figures are means per round over the 3 requests of a cell (the first round of a request is kept).
wall   = the engine's own schedule; the time a sync waits is the GPU work queued before it.
phases = a sync after every phase; each phase's number is its own GPU time plus one sync.
"""
import json
import statistics
import sys
from collections import defaultdict

rounds = [json.loads(line) for line in open(sys.argv[1])]
runs = [json.loads(line) for line in open(sys.argv[2])]
runs = [r for r in runs if r["mode"] == "measured"]
measured_tags = {r["tag"] for r in runs}
rounds = [r for r in rounds if r["tag"] in measured_tags]
unprofiled = sum(1 for r in rounds if r["mode"] not in ("wall", "phases"))
rounds = [r for r in rounds if r["mode"] in ("wall", "phases")]

cells = defaultdict(list)
for r in rounds:
    kind, prof, sampling = r["tag"].split("/")[:3]
    cells[(kind, sampling, prof)].append(r)

mean = lambda xs: sum(xs) / len(xs) if xs else 0.0


def f(rs, key):
    return mean([r.get(key) or 0.0 for r in rs])


def extra_positions(r):
    return sum(p["build"] + p["eval"] + p["tolist"] for p in r.get("positions", [])[1:])


def verify_wait(r):
    """wall: where the verify forward's GPU time is waited for."""
    if r["greedy"]:
        return r["sync_target_tolist_ms"]
    p0 = r["positions"][0]
    return p0["eval"]


def first_position_host(r):
    if r["greedy"]:
        return r.get("walk_host_ms", 0.0)
    p0 = r["positions"][0]
    return p0["build"] + p0["tolist"] + r["lse_build_ms"]


print("## Requests (decode tok/s from LmkChatDone; rounds counted by the profiler)\n")
print("| kind | sampling | profile | decode tok/s median (min–max) | rounds / request | round ms = decode ms / rounds | accepted / drafted |")
print("|---|---|---|---|---|---|---|")
per_tag = defaultdict(int)
for r in rounds:
    per_tag[r["tag"]] += 1
groups = defaultdict(list)
for r in runs:
    groups[(r["kind"], r["sampling"], r["prof"])].append(r)
for key in sorted(groups):
    rs = groups[key]
    t = sorted(x["decode_tps"] for x in rs)
    n = [per_tag[x["tag"]] for x in rs]
    rm = [(x["done"]["totalMs"] - x["done"]["ttftMs"]) / per_tag[x["tag"]] for x in rs if per_tag[x["tag"]]]
    acc = sum(x["done"].get("draftAccepted") or 0 for x in rs)
    dr = sum(x["done"].get("draftDrafted") or 0 for x in rs)
    print(f"| {key[0]} | {key[1]} | {key[2]} | {statistics.median(t):.1f} ({t[0]:.1f}–{t[-1]:.1f}) | {mean(n):.0f} | "
          f"{statistics.median(rm):.1f} | {acc} / {dr} ({acc / dr * 100:.0f}%) |")

print("\n## Wall profile: one round in the engine's own schedule (ms per round, mean)\n")
print("| kind | sampling | rounds | verify positions | walked positions | step (next()) | gap between steps | round | "
      "draft build | draft wait (1st sync) | verify build | verify wait | walk host (pos 1 build/tolist, lse build) | "
      "positions 2.. (build+eval+tolist) | bookkeeping | rollback build | rest |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(k for k in cells if k[2] == "wall"):
    rs = cells[key]
    parts = {
        "draft_build": f(rs, "draft_build_ms"), "draft_wait": f(rs, "sync_draft_tolist_ms"),
        "verify_build": f(rs, "verify_build_ms"), "verify_wait": mean([verify_wait(r) for r in rs]),
        "walk_host": mean([first_position_host(r) for r in rs]), "extra": mean([extra_positions(r) for r in rs]),
        "book": f(rs, "bookkeeping_ms"), "rb": f(rs, "rollback_build_ms"),
    }
    rnd = f(rs, "round_ms")
    rest = rnd - sum(parts.values())
    print(f"| {key[0]} | {key[1]} | {len(rs)} | {mean([r['block'] for r in rs]):.2f} | {f(rs, 'walk_positions'):.2f} | "
          f"{f(rs, 'step_ms'):.1f} | {mean([r['gap_before_ms'] or 0 for r in rs]):.1f} | {rnd:.1f} | "
          + " | ".join(f"{v:.2f}" for v in parts.values()) + f" | {rest:.2f} |")

print("\n## Phases profile: a sync after every phase (ms per round, mean; each = that phase's GPU time + one sync)\n")
print("| kind | sampling | rounds | draft | verify | argmax (greedy) | logsumexp (sampled) | per walked position: build / eval / tolist "
      "| walk total | rollback | round |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(k for k in cells if k[2] == "phases"):
    rs = cells[key]
    pos = [p for r in rs for p in r.get("positions", [])]
    pp = (f"{mean([p['build'] for p in pos]):.2f} / {mean([p['eval'] for p in pos]):.2f} / {mean([p['tolist'] for p in pos]):.2f}"
          if pos else "—")
    print(f"| {key[0]} | {key[1]} | {len(rs)} | {f(rs, 'draft_ms'):.2f} | {f(rs, 'verify_ms'):.2f} | "
          f"{f(rs, 'argmax_ms'):.2f} | {f(rs, 'lse_ms'):.2f} | {pp} | {f(rs, 'walk_ms'):.2f} | {f(rs, 'rollback_ms'):.2f} | "
          f"{f(rs, 'round_ms'):.1f} |")

print("\n## Micro-benchmark on real 32k verify logits (ms, median of per-round medians; 5 reps each after a warm call)\n")
micro = defaultdict(list)
for r in rounds:
    if "micro" in r:
        micro[(r["tag"].split("/")[0], "sampled" if not r["greedy"] else "greedy")].append(r["micro"])
names = ["positions", "per_position_all", "vectorized_all", "argmax_block", "lse_one", "top_p_one", "top_k_one",
         "categorical_one", "sync_only"]
print("| logits from | samples | " + " | ".join(names) + " |")
print("|---|---|" + "---|" * len(names))
for key in sorted(micro):
    ms = micro[key]
    print(f"| {key[0]} {key[1]} | {len(ms)} | " + " | ".join(f"{statistics.median(m[n] for m in ms):.2f}" for n in names) + " |")
print(f"\nrounds the profiler could not time (logits processors or a positioned sampler): {unprofiled}")
print("\nper_position_all = the engine's walk over every verify position (logsumexp stack, then per position: sampler, "
      "mx.eval, tolist); vectorized_all = one logsumexp over the block, the sampler on [positions, vocab], one eval, "
      "one tolist; the *_one rows are the pieces for one position; sampler = temp 1.0 / top_p 0.95 / top_k 20.")
