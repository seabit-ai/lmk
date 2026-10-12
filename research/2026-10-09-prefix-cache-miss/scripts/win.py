import json, sys, datetime
for line in open(sys.argv[1]):
    try: d = json.loads(line)
    except: continue
    t = d.get("time_ms", 0)
    if not (1791586230000 <= t <= 1791586760000): continue
    ts = datetime.datetime.fromtimestamp(t/1000).strftime("%H:%M:%S")
    ev = d.get("event")
    keep = {k: d.get(k) for k in ("refId","purpose","promptTokens","cachedTokens","outcome","msg","restoredTokens","yieldedAtTokens","totalMs","cancelled") if k in d}
    print(ts, ev, keep)
