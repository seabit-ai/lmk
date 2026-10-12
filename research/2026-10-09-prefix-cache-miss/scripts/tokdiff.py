import json, sys, glob
from tokenizers import Tokenizer
snap = glob.glob(sys.argv[3] + "/snapshots/*/tokenizer.json")[0]
tok = Tokenizer.from_file(snap)
def sysc(p):
    with open(p) as f:
        return json.load(f)["body"]["messages"][0]["content"]
a, b = sysc(sys.argv[1]), sysc(sys.argv[2])
ta, tb = tok.encode(a).ids, tok.encode(b).ids
n = 0
while n < min(len(ta), len(tb)) and ta[n] == tb[n]:
    n += 1
print(f"sys tokens: A={len(ta)} B={len(tb)}, common prefix={n}")
# char position of divergence
import difflib
i = 0
while i < min(len(a), len(b)) and a[i] == b[i]:
    i += 1
print(f"char common prefix={i} of lenA={len(a)} lenB={len(b)}")
# does the remainder realign after B's insertion?
rest_a, rest_b = a[i:], b[i:]
# find where rest_a reappears in rest_b
idx = rest_b.find(rest_a[:200])
print(f"A-remainder reappears in B at offset: {idx} (insertion size)")
