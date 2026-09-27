"""exp01: with a DFlash drafter the prefill asks for captured layers, which sends a continued (restored) prompt
through mlx-vlm's original call instead of the fork's text fast path. Same tokens, same split, both paths."""
import sys
import uuid

sys.path.insert(0, "tests")
from itest_model import model_dir  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_vlm.models.cache import make_prompt_cache  # noqa: E402
from lmk.engine import MlxEngine  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "file_read", "description": "Read a text file and return its contents.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}},
                   "required": ["path"]}}}]
engine = MlxEngine("exp", model_dir(), 32768)
lm = engine._kit.model.language_model
fmt = engine.chat_format()


def last(tokens, splits, **kw_last):
    cache = make_prompt_cache(lm)
    start = 0
    bounds = list(splits) + [len(tokens)]
    for k, end in enumerate(bounds):
        kw = kw_last if k == len(bounds) - 1 else {}
        out = lm(mx.array([tokens[start:end]]), cache=cache, **kw)
        start = end
    lg = out.logits[0, -1].astype(mx.float32)
    return lg - mx.logsumexp(lg)


def report(label, a, b):
    import math
    top = mx.argsort(-a)[:20]
    pa, pb = mx.exp(a), mx.exp(b)
    tv = 0.5 * mx.abs(pa - pb).sum().item()
    big = [abs(a[t].item() - b[t].item()) for t in top.tolist() if max(a[t].item(), b[t].item()) > math.log(0.05)]
    print(f"   {label}: TV {tv:.3f} max|dlogprob| (p>0.05) {max(big):.3f}", flush=True)


for trial in range(3):
    prompt = fmt.render([{"role": "system", "content": f"Session {uuid.uuid4().hex}. You are a creative assistant."},
                         {"role": "user", "content": "Invent a name for a new colour and describe it in two sentences."}],
                        TOOLS)
    tokens = engine.preflight(prompt).tokens
    one = last(tokens, [])
    print(f"trial {trial} n={len(tokens)}", flush=True)
    report("one call vs 256+rest (fast path)", one, last(tokens, [256]))
    report("one call vs 256+rest (rest with capture_layer_ids)", one, last(tokens, [256], capture_layer_ids=[1]))
    report("one call vs one call with capture_layer_ids", one, last(tokens, [], capture_layer_ids=[1]))
