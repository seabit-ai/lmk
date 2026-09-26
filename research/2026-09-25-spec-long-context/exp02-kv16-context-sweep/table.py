"""kv8 (exp01) next to kv16 (exp02): per-step ms, tok/s, speedup, acceptance, rounds, memory.

usage: table.py <exp01 runs.jsonl> <exp02 runs.jsonl>
Round ms: exp02 counts rounds exactly (the temporary lmk's draft.rounds before/after each request);
exp01 had no round count, so its round ms is the estimate from exp01's results (255 - accepted rounds).
"""
import json
import statistics
import sys
from collections import defaultdict

KV16_BYTES_PER_TOKEN = 65536   # engine context-fit log, exp02 raw/context-fit-*.txt (full_kv=…B/token)
KV8_BYTES_PER_TOKEN = 34816    # same log line in exp01 raw/serve-off.log


def load(path):
    rows, discarded = defaultdict(list), 0
    for line in open(path):
        r = json.loads(line)
        if r["mode"] == "discarded":
            discarded += 1
        elif r["mode"] != "warm":
            rows[(r["ctx"], r["kind"], r["mode"], r["cond"])].append(r)
    return rows, discarded


kv8, d8 = load(sys.argv[1])
kv16, d16 = load(sys.argv[2])
med = lambda rs, f: statistics.median(f(x) for x in rs)
tps = lambda rs: med(rs, lambda x: x["decode_tps"])
ms_step = lambda rs: 1000 / tps(rs)
rng = lambda rs: f"{min(x['decode_tps'] for x in rs):.1f}–{max(x['decode_tps'] for x in rs):.1f}"


def acc(rs):
    a = sum(x["done"].get("draftAccepted") or 0 for x in rs)
    d = sum(x["done"].get("draftDrafted") or 0 for x in rs)
    return a, d


def round_ms(rs, exact):
    if exact and all("rounds" in x and x["rounds"] for x in rs):
        return med(rs, lambda x: (x["done"]["totalMs"] - x["done"]["ttftMs"]) / x["rounds"])
    return med(rs, lambda x: (x["done"]["totalMs"] - x["done"]["ttftMs"]) / max(1, 255 - (x["done"].get("draftAccepted") or 0)))


keys = sorted({k[:3] for k in kv16}, key=lambda k: (k[2], k[0], k[1]))
print("## Decode: kv8 (exp01) vs kv16 (exp02)\n")
print("| context | kind | mode | off ms/step kv8 → kv16 | off tok/s kv8 → kv16 (kv16 min–max) | on tok/s kv8 → kv16 (kv16 min–max) "
      "| speedup kv8 → kv16 | accepted kv8 → kv16 | round ms kv8 (est.) → kv16 (exact) | greedy text off = on, kv16 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for key in keys:
    o8, n8, o16, n16 = kv8.get(key + ("off",)), kv8.get(key + ("on",)), kv16.get(key + ("off",)), kv16.get(key + ("on",))
    if not (o8 and n8 and o16 and n16):
        continue
    a8, dr8 = acc(n8)
    a16, dr16 = acc(n16)
    same = "—"
    if key[2] == "greedy":
        so, sn = {x["sha"] for x in o16}, {x["sha"] for x in n16}
        same = "yes" if so == sn and len(so) == 1 else f"no ({len(so)} off / {len(sn)} on distinct)"
    print(f"| {key[0] // 1024}k | {key[1]} | {key[2]} | {ms_step(o8):.1f} → {ms_step(o16):.1f} | "
          f"{tps(o8):.1f} → {tps(o16):.1f} ({rng(o16)}) | {tps(n8):.1f} → {tps(n16):.1f} ({rng(n16)}) | "
          f"{tps(n8) / tps(o8):.2f}× → {tps(n16) / tps(o16):.2f}× | {a8 / dr8 * 100:.0f}% → {a16 / dr16 * 100:.0f}% | "
          f"{round_ms(n8, False):.0f} → {round_ms(n16, True):.0f} | {same} |")

print("\n## Memory (kv16 measured; kv8 derived)\n")
print("| context (prompt tokens) | KV bytes kv8 → kv16 (engine bytes/token × tokens) | kv16 lmk_gpu_bytes max during decode, off / on "
      "| kv8 derived = kv16 − KV difference, off |")
print("|---|---|---|---|")
for ctx in sorted({k[0] for k in kv16}):
    o16 = kv16.get((ctx, "code", "greedy", "off")) + kv16.get((ctx, "prose", "greedy", "off"))
    n16 = kv16.get((ctx, "code", "greedy", "on")) + kv16.get((ctx, "prose", "greedy", "on"))
    tokens = max(x["done"]["promptTokens"] + x["done"]["completionTokens"] for x in o16)
    g_off = max(x["gpu_bytes_max"] for x in o16)
    g_on = max(x["gpu_bytes_max"] for x in n16)
    diff = (KV16_BYTES_PER_TOKEN - KV8_BYTES_PER_TOKEN) * tokens
    print(f"| {ctx // 1024}k ({tokens}) | {KV8_BYTES_PER_TOKEN * tokens / 1e9:.2f} → {KV16_BYTES_PER_TOKEN * tokens / 1e9:.2f} GB | "
          f"{g_off / 1e9:.2f} / {g_on / 1e9:.2f} GB | {(g_off - diff) / 1e9:.2f} GB |")
print(f"\nruns discarded because the resident lmk was busy: exp01 {d8}, exp02 {d16}")
