"""exp03 run2: is the check's per-position logprob the model's own teacher-forced logprob? A reference forward
of the same prompt outside the engine (fresh cache, one call), logprob of token t+1 at position t."""
import sys
import uuid

sys.path.insert(0, "tests")
sys.path.insert(0, "research/2026-09-26-context-check/exp03-restore-vs-cold")
from itest_model import model_dir  # noqa: E402

import mlx.core as mx  # noqa: E402
import mlx_engine.model_kit.batched_vision.context_check as cc  # noqa: E402
import mlx_engine.model_kit.batched_vision.model_kit as vision_kit  # noqa: E402
from mlx_vlm.models.cache import make_prompt_cache  # noqa: E402
from lmk.engine import MlxEngine  # noqa: E402

orig_summary = cc.ContextCheck.summary
cc.ContextCheck.summary = lambda self: {**orig_summary(self), "raw": list(self.logprobs)}
engine = MlxEngine("exp", model_dir(), 32768)
fmt = engine.chat_format()
lm = engine._kit.model.language_model

text = fmt.render([{"role": "system", "content": f"Session {uuid.uuid4().hex}."},
                   {"role": "user", "content": "Count: " + ", ".join(str(i) for i in range(1, 200))}], None)
tokens = engine.preflight(text).tokens


def reference(toks):
    cache = make_prompt_cache(lm)
    out = lm(mx.array([toks]), cache=cache)
    logits = out.logits[0].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = mx.array(toks[1:])
    same = mx.take_along_axis(lp[:-1], tgt[:, None], axis=-1)[:, 0]          # P(token t+1 | <= t)
    return same.tolist()


ref = reference(tokens)
g = engine.generate(text, max_tokens=1, request_id="exp-ref", on_prefill=lambda *a: True, sampling={"temp": 0.0},
                    tokens=tokens)
"".join(g)
raw = g.stats.context_check["raw"]
k = len(raw)
d = [abs(a - b) for a, b in zip(raw, ref[-k:])]
print(f"prompt={len(tokens)} scored={k} check mean surprise={-sum(raw) / k:.3f} reference={-sum(ref[-k:]) / k:.3f} "
      f"max|d|={max(d):.3f} mean|d|={sum(d) / k:.4f}")
shift = [abs(a - b) for a, b in zip(raw[1:], ref[-k:-1])]
print(f"shifted by one: mean|d|={sum(shift) / len(shift):.4f}")
print("check first 8:", [round(x, 2) for x in raw[:8]])
print("ref   same 8: ", [round(x, 2) for x in ref[-k:][:8]])
