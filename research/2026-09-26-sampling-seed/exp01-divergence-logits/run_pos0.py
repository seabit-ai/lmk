"""exp01: the first token's distribution (the prefill's last logits), cold vs restored from disk at 256, in the
engine, with and without a DFlash drafter loaded. argv[1]: draft | nodraft"""
import math
import sys
import uuid

sys.path.insert(0, "tests")
from itest_model import model_dir  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_engine.utils import sampling as eng_sampling  # noqa: E402
from lmk.engine import MlxEngine  # noqa: E402

DRAFT = "/Users/xinkai/.cache/huggingface/hub/models--seabit-ai--Qwen3.8-27B-DFlash2-4bit/snapshots/local-quant-2026-09-24"
TOOLS = [{"type": "function", "function": {
    "name": "file_read", "description": "Read a text file and return its contents.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}},
                   "required": ["path"]}}}]
seen = {}
orig = eng_sampling.SeededSampler.sample_target


def traced(self, logprobs, *, row_ids, positions):
    if 0 in [int(p) for p in positions]:
        seen["lp"] = (logprobs if logprobs.ndim == 2 else logprobs[None])[0].astype(mx.float32).tolist()
    return orig(self, logprobs, row_ids=row_ids, positions=positions)


eng_sampling.SeededSampler.sample_target = traced
engine = MlxEngine("exp", model_dir(), 32768, draft_path=DRAFT if sys.argv[1] == "draft" else None)
fmt = engine.chat_format()


def first(prompt):
    g = engine.generate(prompt, max_tokens=1, request_id=f"exp-{uuid.uuid4().hex[:8]}", on_prefill=lambda *a: True,
                        sampling={"temp": 1.0, "top_p": 0.95, "top_k": 20, "seed": 1})
    "".join(g)
    return seen["lp"], g.stats.cached_tokens


def diff(a, b):
    a, b = mx.array(a), mx.array(b)
    top = mx.argsort(-a)[:20].tolist()
    tv = 0.5 * mx.abs(mx.exp(a) - mx.exp(b)).sum().item()
    big = max(abs(a[t].item() - b[t].item()) for t in top if max(a[t].item(), b[t].item()) > math.log(0.05))
    return f"TV {tv:.3f} max|dlogprob| (p>0.05) {big:.3f}"


from mlx_vlm.models.cache import make_prompt_cache  # noqa: E402

lm = engine._kit.model.language_model


def reference(prompt):
    tokens = engine.preflight(prompt).tokens
    out = lm(mx.array([tokens]), cache=make_prompt_cache(lm))
    lg = out.logits[0, -1].astype(mx.float32)
    return (lg - mx.logsumexp(lg)).tolist()


for tools in (True, False):
    for trial in range(3):
        prompt = fmt.render([{"role": "system", "content": f"Session {uuid.uuid4().hex}. You are a creative assistant. "
                              + " ".join(f"Rule {i}: be vivid." for i in range(0 if tools else 60))},
                             {"role": "user", "content": "Invent a name for a new colour and describe it in two sentences."}],
                            TOOLS if tools else None)
        ref = reference(prompt)
        cold, c0 = first(prompt)
        restored, c1 = first(prompt)
        again, c2 = first(prompt)
        print(f"{sys.argv[1]} tools={tools} trial {trial}: cached {c0}/{c1}/{c2}; cold vs restored {diff(cold, restored)}; "
              f"restored vs again {diff(restored, again)}; reference (one plain call) vs cold {diff(ref, cold)}, "
              f"vs restored {diff(ref, restored)}", flush=True)
