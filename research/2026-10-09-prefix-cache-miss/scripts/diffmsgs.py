import json, sys, hashlib
def msgs(p):
    with open(p) as f:
        return json.load(f)["body"]["messages"]
def canon(m):
    return json.dumps(m, sort_keys=True, ensure_ascii=False)
a, b = msgs(sys.argv[1]), msgs(sys.argv[2])
print(f"counts: A={len(a)} B={len(b)}")
chars = 0
for i in range(min(len(a), len(b))):
    ca, cb = canon(a[i]), canon(b[i])
    if ca != cb:
        print(f"FIRST DIFF at message {i} (role A={a[i].get('role')} B={b[i].get('role')}), ~{chars} chars into prompt")
        print(f"  lenA={len(ca)} lenB={len(cb)}")
        # find first differing char
        for j in range(min(len(ca), len(cb))):
            if ca[j] != cb[j]:
                print(f"  first char diff at {j}:")
                print("  A:", ca[max(0,j-120):j+200])
                print("  B:", cb[max(0,j-120):j+200])
                break
        else:
            print("  one is a prefix of the other")
            print("  A tail:", ca[len(cb):][:300] if len(ca)>len(cb) else "")
            print("  B tail:", cb[len(ca):][:300] if len(cb)>len(ca) else "")
        sys.exit(0)
    chars += len(ca)
print(f"first {min(len(a),len(b))} messages identical ({chars} chars)")
for i in range(min(len(a), len(b)), max(len(a), len(b))):
    m = (a if len(a)>len(b) else b)[i]
    print(f"  extra message {i}: role={m.get('role')} len={len(canon(m))}")
