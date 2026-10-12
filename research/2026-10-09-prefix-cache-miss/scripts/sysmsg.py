import json, sys, hashlib
for p in sys.argv[1:]:
    try:
        with open(p) as f:
            m = json.load(f)["body"]["messages"][0]
        c = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"], ensure_ascii=False)
        has = "Browser identities this repo may use" in c
        print(f"{p.split('/')[-1]}  sys_len={len(c)}  md5={hashlib.md5(c.encode()).hexdigest()[:8]}  identities={'YES' if has else 'no'}")
    except Exception as e:
        print(f"{p.split('/')[-1]}  ERROR {e}")
