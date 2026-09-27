import sys
sys.path.insert(0, "tests")
from itest_model import model_dir
import mlx.core as mx
from lmk.engine import MlxEngine
from mlx_vlm.models.cache import make_prompt_cache
engine = MlxEngine("exp", model_dir(), 32768)
lm = engine._kit.model.language_model
tok = engine._kit.tokenizer
fmt = engine.chat_format()
text = fmt.render([{"role": "system", "content": "Session abc."},
                   {"role": "user", "content": "Count: " + ", ".join(str(i) for i in range(1, 40))}], None)
t = engine.preflight(text).tokens
print(repr(text[:200]))
out = lm(mx.array([t]), cache=make_prompt_cache(lm))
lg = out.logits[0].astype(mx.float32)
lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
s = (-mx.take_along_axis(lp[:-1], mx.array(t[1:])[:, None], axis=-1)[:, 0]).tolist()
am = mx.argmax(lg, axis=-1).tolist()
for i in range(len(t) - 1):
    print(i, repr(tok.decode([t[i]])), "->", repr(tok.decode([t[i + 1]])), round(s[i], 2), "argmax", repr(tok.decode([am[i]])))
