"""exp01 control: how much does the last-position logprob move when the same prompt is read in one call vs two
(256 + rest), no disk involved — pure chunking / reduction-order noise — against a disk restore at 256."""
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


def last_logprobs(tokens, splits):
    cache = make_prompt_cache(lm)
    start = 0
    for end in list(splits) + [len(tokens)]:
        out = lm(mx.array([tokens[start:end]]), cache=cache)
        start = end
    lg = out.logits[0, -1].astype(mx.float32)
    return lg - mx.logsumexp(lg), out.logits.dtype


for trial in range(3):
    prompt = fmt.render([{"role": "system", "content": f"Session {uuid.uuid4().hex}. You are a creative assistant."},
                         {"role": "user", "content": "Invent a name for a new colour and describe it in two sentences."}],
                        TOOLS)
    tokens = engine.preflight(prompt).tokens
    one, dtype = last_logprobs(tokens, [])
    for splits in ([256], [128, 256], [len(tokens) - 1]):
        other, _ = last_logprobs(tokens, splits)
        top = mx.argsort(-one)[:20]
        d = mx.abs(one[top] - other[top])
        print(f"trial {trial} n={len(tokens)} logits dtype {dtype}: one call vs splits {splits}: "
              f"max|dlogprob| top-20 {d.max().item():.4f}, top-1 {one[top[0]].item():.3f} / {other[top[0]].item():.3f}",
              flush=True)
    again, _ = last_logprobs(tokens, [])
    print(f"trial {trial}: one call vs one call again: {mx.abs(one - again).max().item():.4f}", flush=True)
