import json, sys
start, end = 1791567120000, 1791586750000  # 10:32 .. just after .1 done
for line in open(sys.argv[1]):
    try: d = json.loads(line)
    except: continue
    if d.get("event") not in ("LmkChatDone", "LmkWarmupDone"): continue
    t = d.get("time_ms", 0)
    if not (start <= t <= end): continue
    ref = d.get("refId") or ""
    if ref.startswith("s-1790466395702/warmup"): continue
    import datetime
    ts = datetime.datetime.fromtimestamp(t/1000).strftime("%H:%M:%S")
    print(f'{ts} {d["event"]:14} {ref:40} prompt={d.get("promptTokens")} cached={d.get("cachedTokens")}')
