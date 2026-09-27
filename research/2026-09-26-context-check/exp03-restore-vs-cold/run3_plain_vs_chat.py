import sys, uuid
sys.path.insert(0, "tests")
from itest_model import model_dir
import mlx.core as mx
from lmk.engine import MlxEngine
from mlx_vlm.models.cache import make_prompt_cache
engine = MlxEngine("exp", model_dir(), 32768)
lm = engine._kit.model.language_model
tok = engine._kit.tokenizer
fmt = engine.chat_format()
for label, text in (("plain", ", ".join(str(i) for i in range(1, 200))),
                    ("chat", fmt.render([{"role": "system", "content": "Session abc."},
                                         {"role": "user", "content": "Count: " + ", ".join(str(i) for i in range(1, 200))}], None))):
    toks = engine.preflight(text).tokens
    for n in (len(toks), 300):
        t = toks[:n]
        out = lm(mx.array([t]), cache=make_prompt_cache(lm))
        lg = out.logits[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        s = -mx.take_along_axis(lp[:-1], mx.array(t[1:])[:, None], axis=-1)[:, 0]
        print(label, "n", n, "mean", round(s.mean().item(), 3), "last100", round(s[-100:].mean().item(), 3))
