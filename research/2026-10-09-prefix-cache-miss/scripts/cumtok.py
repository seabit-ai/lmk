import json, sys, glob
from tokenizers import Tokenizer
snap = glob.glob(sys.argv[2] + "/snapshots/*/tokenizer.json")[0]
tok = Tokenizer.from_file(snap)
msgs = json.load(open(sys.argv[1]))["body"]["messages"]
cum = 0
for i, m in enumerate(msgs[:30]):
    c = m.get("content")
    if not isinstance(c, str):
        c = json.dumps(c, ensure_ascii=False)
    extra = ""
    for k in ("tool_calls","tool_call_id","name"):
        if k in m: extra += f" {k}"
    n = len(tok.encode(c).ids)
    cum += n + 4
    head = c[:70].replace("\n", "\\n")
    print(f"[{i:3}] {m.get('role'):9} +{n:6} cum≈{cum:6}{extra}  {head}")
    if cum > 13000: break
