"""Result table from raw/runs.jsonl: context x kind x mode -> off / on decode tok/s, speedup, acceptance."""
import json
import statistics
import sys
from collections import defaultdict

rows = defaultdict(list)
discarded = 0
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["mode"] == "discarded":
        discarded += 1
    elif r["mode"] != "warm":
        rows[(r["ctx"], r["kind"], r["mode"], r["cond"])].append(r)


def fmt(rs):
    t = sorted(x["decode_tps"] for x in rs)
    return statistics.median(t), f"{statistics.median(t):.1f} ({t[0]:.1f}–{t[-1]:.1f})"


print("| context (prompt tokens) | kind | mode | off tok/s median (min–max) | on tok/s median (min–max) | speedup | "
      "accepted / drafted (rate) | finish:tokens off / on | greedy text off = on |")
print("|---|---|---|---|---|---|---|---|---|")
for key in sorted({k[:3] for k in rows}, key=lambda k: (k[2], k[0], k[1])):
    off, on = rows.get(key + ("off",), []), rows.get(key + ("on",), [])
    if not off or not on:
        continue
    moff, soff = fmt(off)
    mon, son = fmt(on)
    acc = sum(x["done"].get("draftAccepted") or 0 for x in on)
    dr = sum(x["done"].get("draftDrafted") or 0 for x in on)
    fin = lambda rs: ",".join(sorted({f"{x['done']['finishReason']}:{x['done']['completionTokens']}" for x in rs}))
    shas_off, shas_on = {x["sha"] for x in off}, {x["sha"] for x in on}
    same = "—" if key[2] != "greedy" else ("yes" if shas_off == shas_on and len(shas_off) == 1
                                            else f"no ({len(shas_off)} off / {len(shas_on)} on distinct)")
    print(f"| {key[0] // 1024}k ({off[0]['done']['promptTokens']}) | {key[1]} | {key[2]} | {soff} | {son} | "
          f"{mon / moff:.2f}× | {acc} / {dr} ({acc / dr * 100 if dr else 0:.0f}%) | {fin(off)} / {fin(on)} | {same} |")
print(f"\nruns discarded because the resident lmk was busy: {discarded}")
