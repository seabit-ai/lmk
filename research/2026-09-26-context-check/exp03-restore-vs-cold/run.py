"""exp03: per-position logprobs of the same prompt, cold vs restored. See README.md.
PYTHONPATH=.engine/mlx-engine:. .venv/bin/python research/2026-09-26-context-check/exp03-restore-vs-cold/run.py"""
import random
import sys
import uuid

sys.path.insert(0, "tests")
from itest_model import model_dir  # noqa: E402

import mlx.core as mx  # noqa: E402
import mlx_engine.model_kit.batched_vision.context_check as cc  # noqa: E402
import mlx_engine.model_kit.batched_vision.model_kit as vision_kit  # noqa: E402
from lmk.engine import MlxEngine  # noqa: E402

orig_summary = cc.ContextCheck.summary
cc.ContextCheck.summary = lambda self: {**orig_summary(self), "raw": list(self.logprobs)}

engine = MlxEngine("exp", model_dir(), 32768)
fmt = engine.chat_format()
WORDS = ("amber basalt cobalt dune ember fjord garnet harbor indigo jasper kelp lumen marble nectar onyx pewter "
         "quartz russet sable tundra umber velvet willow xenon yarrow zephyr").split()
rng = random.Random(7)
names = [f"{w}{rng.randint(100, 999)}" for w in WORDS]
weeks = [rng.randint(2, 52) for _ in WORDS]


def conversation(nonce, filler):
    history = [{"role": "system", "content": f"Session {nonce}. You are kitten, a coding agent."},
               {"role": "user", "content": "Release plan. Each codename ships in the week given: " +
                " ".join(f"Codename {n} ships in week {w}." for n, w in zip(names, weeks)) + " Remember this plan." +
                "".join(f" Aside {i}: the {WORDS[i % 26]} team met on floor {i * 13 % 29}." for i in range(filler))},
               {"role": "assistant", "content": "Noted — I have the release plan."}]
    turn = history + [{"role": "user", "content": "The scheduler exported the plan. Output of plan_export:\n" +
                       "\n".join(f"{n}: week {w}" for n, w in zip(names, weeks))}]
    return history, turn


def run(messages, hook=None):
    vision_kit.RESTORED_CACHE_HOOK = hook
    try:
        g = engine.generate(fmt.render(messages, None), max_tokens=1, request_id=f"exp-{uuid.uuid4().hex[:8]}",
                            on_prefill=lambda *a: True, sampling={"temp": 0.0})
        "".join(g)
    finally:
        vision_kit.RESTORED_CACHE_HOOK = None
    return g.stats


def zero_recurrent(restored):
    for c in restored.prompt_cache:
        if type(c).__name__ == "ArraysCache":
            c.cache = [None if a is None else mx.zeros_like(a) for a in c.cache]


def compare(label, a, b):
    k = min(len(a), len(b))
    d = [abs(x - y) for x, y in zip(a[-k:], b[-k:])]
    worst = max(range(k), key=d.__getitem__)
    print(f"  {label}: common={k} max|dlogprob|={max(d):.3f} at -{k - worst} mean|d|={sum(d) / k:.4f} "
          f"surprise {-sum(a[-k:]) / k:.3f} vs {-sum(b[-k:]) / k:.3f}", flush=True)


for filler in (0, 40):
    nonce = uuid.uuid4().hex
    history, turn = conversation(nonce, filler)
    cold = run(turn)                               # nothing cached for this nonce yet
    nonce2 = uuid.uuid4().hex
    history2, turn2 = conversation(nonce2, filler)
    run(history2)
    restored = run(turn2)
    nonce3 = uuid.uuid4().hex
    history3, turn3 = conversation(nonce3, filler)
    run(history3)
    restored_c1 = run(turn3, zero_recurrent)
    # a second cold run of a fresh nonce: how much do two cold runs differ (same length prompts)?
    history4, turn4 = conversation(uuid.uuid4().hex, filler)
    cold2 = run(turn4)
    for name, s in (("cold", cold), ("restored", restored), ("restored-C1", restored_c1), ("cold2", cold2)):
        c = s.context_check
        print(f"filler={filler} {name}: prompt={s.prompt_tokens} restored={c['restored_tokens']} "
              f"scored={c['scored_tokens']} mean={c.get('surprise_mean')}", flush=True)
    compare("cold vs cold2", cold.context_check["raw"], cold2.context_check["raw"])
    compare("cold vs restored", cold.context_check["raw"], restored.context_check["raw"])
    compare("cold vs restored-C1", cold.context_check["raw"], restored_c1.context_check["raw"])
