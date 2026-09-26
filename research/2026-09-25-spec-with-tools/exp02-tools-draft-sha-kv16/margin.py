"""exp02 probe: at the first character where draft-on and draft-off differ, how far apart are the
top two next-token logits? One plain full-prompt forward (kv16, no cache), prompt + the shared prefix.
usage: margin.py <results.json> <off run name> <on run name>"""
import json, sys
import mlx.core as mx
sys.path.insert(0, "research/2026-09-25-spec-with-tools/exp01-tools-draft-sha")
from run import MESSAGES, TOOLS
from lmk.config import load_config
from lmk.engine import MlxEngine
from lmk.models import resolve_model

res = json.load(open(sys.argv[1]))
runs = {r["name"]: r for r in res["runs"]}
off, on = runs[sys.argv[2]]["text"], runs[sys.argv[3]]["text"]
i = next(k for k, (a, b) in enumerate(zip(off, on)) if a != b)
model = resolve_model(load_config().model.source)
engine = MlxEngine("probe", model.path, 32768, template_kwargs={"enable_thinking": True, "reasoning_effort": "low"})
kit = engine._kit
tok = kit.tokenizer
prompt = engine.chat_format().render(MESSAGES, TOOLS)
# back up to a token boundary both texts share: the longest prefix whose tokens re-decode unchanged
prefix = off[:i]
ids = tok.encode(prompt + prefix, add_special_tokens=False)
ids = ids[:-1]  # the last token may straddle the difference
shared = tok.decode(ids[len(tok.encode(prompt, add_special_tokens=False)):])
lm = getattr(kit.model, "language_model", kit.model)
out = lm(mx.array([ids]))
logits = (out.logits if hasattr(out, "logits") else out)[0, -1].astype(mx.float32)
order = mx.argsort(-logits)[:5].tolist()
print("difference at char", i, "shared prefix ends", repr(shared[-60:]))
print("off continues", repr(off[len(shared):len(shared) + 30]), "| on continues", repr(on[len(shared):len(shared) + 30]))
for t in order:
    print(f"  {logits[t].item():9.4f}  {tok.decode([t])!r}")
engine.close()
