import json, sys, hashlib, os, datetime
sys.path.insert(0, "/Users/xinkai/src/lmk")
from transformers import AutoTokenizer
import glob
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--lmstudio-community--Qwen3.8-27B-MLX-4bit/snapshots/*"))[0]
tok = AutoTokenizer.from_pretrained(snap)
from lmk.chatformat import QWEN

CACHE = os.path.expanduser("~/.lmk/cache/d3caf54bed4bc104e7ef91e9")
_INITIAL = hashlib.sha256(b"prompt-prefix-v1").hexdigest()

def render_tokens(path):
    body = json.load(open(path))["body"]
    text = tok.apply_chat_template(QWEN.for_template(body["messages"]), tools=body.get("tools") or None,
                                   tokenize=False, add_generation_prompt=True, reasoning_effort="low")
    return tok.encode(text, add_special_tokens=False)

def chunk_keys(ids, n):
    keys, prev = [], _INITIAL
    for c in range(n):
        toks = ids[c*256:(c+1)*256]
        if len(toks) < 256: break
        key = hashlib.sha256(f"{prev}|{','.join(map(str, toks))}|".encode()).hexdigest()
        keys.append(key)
        prev = key
    return keys

def stat_chain(label, keys):
    print(f"--- {label}")
    for i, k in enumerate(keys, 1):
        row = []
        for kind in ("kv_delta", "state_checkpoint"):
            p = f"{CACHE}/record@{k}@{kind}.safetensors"
            if os.path.exists(p):
                st = os.stat(p)
                b = datetime.datetime.fromtimestamp(st.st_birthtime).strftime("%m-%d %H:%M:%S")
                m = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%m-%d %H:%M:%S")
                row.append(f"{kind}: born {b} touched {m}")
        if row:
            print(f"  chunk {i:3} end={i*256:6}  " + " | ".join(row))

miss = render_tokens(sys.argv[1])
warm = render_tokens(sys.argv[2])
print(f"tokens: miss={len(miss)} (log said 82292)  warmup={len(warm)} (log said 82169)")
n = 0
while n < min(len(miss), len(warm)) and miss[n] == warm[n]: n += 1
print(f"common token prefix: {n} (chunk {n//256})")
mk = chunk_keys(miss, 50)
stat_chain("MISS-variant chain (first 50 chunks)", mk)
