"""exp04: context check split by the role of the predicted token. See README.md.
PYTHONPATH=.engine/mlx-engine:. .venv/bin/python research/2026-09-26-context-check/exp04-by-role/run.py"""
import random
import sys
import uuid

sys.path.insert(0, "tests")
from itest_model import model_dir  # noqa: E402

import mlx.core as mx  # noqa: E402
import mlx_engine.model_kit.batched_vision.context_check as cc  # noqa: E402
import mlx_engine.model_kit.batched_vision.model_kit as vision_kit  # noqa: E402
from lmk.engine import MlxEngine  # noqa: E402

cc.TAIL_TOKENS = 4096
orig_summary = cc.ContextCheck.summary
cc.ContextCheck.summary = lambda self: {**orig_summary(self), "raw": list(self.logprobs)}

engine = MlxEngine("exp", model_dir(), 32768)
fmt = engine.chat_format()
tok = engine._kit.tokenizer
IM_START = tok.encode("<|im_start|>")[-1]
IM_END = tok.encode("<|im_end|>")[-1]
ASSISTANT = tok.encode("assistant")
WORDS = ("amber basalt cobalt dune ember fjord garnet harbor indigo jasper kelp lumen marble nectar onyx pewter "
         "quartz russet sable tundra umber velvet willow xenon yarrow zephyr").split()


def plan(seed):
    rng = random.Random(seed)
    return [f"{w}{rng.randint(100, 999)}" for w in WORDS], [rng.randint(2, 52) for _ in WORDS]


def conversation(nonce, seed):
    names, weeks = plan(seed)
    history = [{"role": "system", "content": f"Session {nonce}. You are kitten, a coding agent."},
               {"role": "user", "content": "Release plan. Each codename ships in the week given: " +
                " ".join(f"Codename {n} ships in week {w}." for n, w in zip(names, weeks)) + " Remember this plan."},
               {"role": "assistant", "content": "Noted — I have the release plan."},
               {"role": "user", "content": " ".join(f"Aside {i}: the {WORDS[i % 26]} team met on floor {i * 13 % 29}."
                                                    for i in range(40))},
               {"role": "assistant", "content": "OK."}]
    turn = history + [{"role": "user", "content": "List the plan, one line per codename."},
                      {"role": "assistant", "content": "\n".join(f"{n}: week {w}" for n, w in zip(names, weeks))},
                      {"role": "user", "content": "Thanks."}]
    return history, turn


def roles(tokens):
    """role of each token's message: 'assistant' or 'other' (template tokens count with their message)."""
    out, role, i = [], "other", 0
    while i < len(tokens):
        if tokens[i] == IM_START:
            role = "assistant" if tokens[i + 1: i + 1 + len(ASSISTANT)] == ASSISTANT else "other"
        out.append(role)
        i += 1
    return out


def run(messages, hook=None):
    text = fmt.render(messages, None)
    tokens = engine.preflight(text).tokens
    vision_kit.RESTORED_CACHE_HOOK = hook
    try:
        g = engine.generate(text, max_tokens=1, request_id=f"exp-{uuid.uuid4().hex[:8]}", tokens=tokens,
                            on_prefill=lambda *a: True, sampling={"temp": 0.0})
        "".join(g)
    finally:
        vision_kit.RESTORED_CACHE_HOOK = None
    return tokens, g.stats


def zero_recurrent(restored):
    for c in restored.prompt_cache:
        if type(c).__name__ == "ArraysCache":
            c.cache = [None if a is None else mx.zeros_like(a) for a in c.cache]


def zero_all(restored):
    zero_recurrent(restored)
    for c in restored.prompt_cache:
        if type(c).__name__ != "ArraysCache" and getattr(c, "keys", None) is not None:
            k, v = c.state
            c.state = (mx.zeros_like(k), mx.zeros_like(v))


captured = {}


def capture(restored):
    captured["len"] = restored.cached_prefix_len
    captured["states"] = [[None if x is None else mx.array(x) for x in c.state] for c in restored.prompt_cache]
    mx.eval([x for s in captured["states"] for x in s if x is not None])


def swap_in(restored):
    if captured.get("len") != restored.cached_prefix_len:
        raise RuntimeError(f"wrong-conversation state covers {captured.get('len')}, this restore {restored.cached_prefix_len}")
    for c, state in zip(restored.prompt_cache, captured["states"]):
        copy = [None if x is None else mx.array(x) for x in state]
        c.state = copy if type(c).__name__ == "ArraysCache" else tuple(copy)


def report(label, tokens, stats):
    c = stats.context_check
    raw = c["raw"]
    seg = c["segment_tokens"]
    targets = roles(tokens)[len(tokens) - seg + 1:]           # role of each scored target token
    targets = targets[len(targets) - len(raw):]
    by = {}
    for r, lp in zip(targets, raw):
        by.setdefault(r, []).append(-lp)
    parts = " ".join(f"{r}: n={len(v)} mean={sum(v) / len(v):.3f}" for r, v in sorted(by.items()))
    print(f"{label}: prompt={len(tokens)} restored={c['restored_tokens']} scored={len(raw)} | {parts}", flush=True)


for repeat in range(2):
    for label, hook in (("intact", None), ("C1-zero-recurrent", zero_recurrent), ("C3-zero-all", zero_all),
                        ("C4-wrong-conversation", swap_in)):
        nonce = uuid.uuid4().hex
        history, turn = conversation(nonce, seed=1)
        if label.startswith("C4"):
            other_history, other_turn = conversation(nonce[::-1], seed=2)   # same shape, another plan
            run(other_history)
            vision_kit_hook = capture
            run(other_turn, vision_kit_hook)
        run(history)
        tokens, stats = run(turn, hook)
        report(f"{label}#{repeat}", tokens, stats)
