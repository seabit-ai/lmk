"""exp02: does the context check see a lost context? See README.md.
PYTHONPATH=.engine/mlx-engine:. .venv/bin/python research/2026-09-26-context-check/exp02-corruption/run.py"""
import random
import sys
import uuid

sys.path.insert(0, "tests")
from itest_model import model_dir  # noqa: E402

import mlx.core as mx  # noqa: E402
import mlx_engine.model_kit.batched_vision.model_kit as vision_kit  # noqa: E402
from lmk.engine import MlxEngine  # noqa: E402

engine = MlxEngine("exp", model_dir(), 32768)
fmt = engine.chat_format()
WORDS = ("amber basalt cobalt dune ember fjord garnet harbor indigo jasper kelp lumen marble nectar onyx pewter "
         "quartz russet sable tundra umber velvet willow xenon yarrow zephyr").split()
rng = random.Random(7)
names = [f"{w}{rng.randint(100, 999)}" for w in WORDS]
weeks = [rng.randint(2, 52) for _ in WORDS]


def conversation(nonce):
    history = [{"role": "system", "content": f"Session {nonce}. You are kitten, a coding agent."},
               {"role": "user", "content": "Release plan. Each codename ships in the week given: " +
                " ".join(f"Codename {n} ships in week {w}." for n, w in zip(names, weeks)) +
                " Remember this plan."},
               {"role": "assistant", "content": "Noted — I have the release plan."}]
    turn = history + [{"role": "user", "content": "The scheduler exported the plan. Output of plan_export:\n" +
                       "\n".join(f"{n}: week {w}" for n, w in zip(names, weeks))}]
    return history, turn


def unrelated(nonce):
    text = " ".join(f"Paragraph {i}: the harbour crane lifted crate {i * 37 % 101} onto the barge at dawn."
                    for i in range(60))
    history = [{"role": "system", "content": f"Session {nonce}. You are kitten, a coding agent."},
               {"role": "user", "content": text}, {"role": "assistant", "content": "Noted."}]
    return history, history + [{"role": "user", "content": "Summarise."}]


def run(messages):
    g = engine.generate(fmt.render(messages, None), max_tokens=1, request_id=f"exp-{uuid.uuid4().hex[:8]}",
                        on_prefill=lambda *a: True, sampling={"temp": 0.0})
    "".join(g)
    return g.stats


def kinds(restored):
    for c in restored.prompt_cache:
        yield type(c).__name__, c


def zero_recurrent(restored):
    for kind, c in kinds(restored):
        if kind == "ArraysCache":
            c.cache = [None if a is None else mx.zeros_like(a) for a in c.cache]


def shuffle_kv(restored):
    zero_recurrent(restored)
    for kind, c in kinds(restored):
        if kind != "ArraysCache" and getattr(c, "keys", None) is not None:
            k, v = c.state
            perm = mx.array(random.Random(1).sample(range(k.shape[2]), k.shape[2]))
            c.state = (k[:, :, perm], v[:, :, perm])


def zero_all(restored):
    zero_recurrent(restored)
    for kind, c in kinds(restored):
        if kind != "ArraysCache" and getattr(c, "keys", None) is not None:
            k, v = c.state
            c.state = (mx.zeros_like(k), mx.zeros_like(v))


captured = {}


def capture(restored):
    captured["len"] = restored.cached_prefix_len
    captured["states"] = [[mx.array(x) if x is not None else None for x in c.state] for c in restored.prompt_cache]
    mx.eval([x for s in captured["states"] for x in s if x is not None])


def swap_in(restored):
    assert captured["len"] == restored.cached_prefix_len, (captured["len"], restored.cached_prefix_len)
    for c, state in zip(restored.prompt_cache, captured["states"]):
        c.state = [mx.array(x) if x is not None else None for x in state] if type(c).__name__ == "ArraysCache" \
            else tuple(mx.array(x) for x in state)


def trial(label, hook=None, before=None):
    history, turn = conversation(uuid.uuid4().hex)
    run(history)
    if before:
        before()
    vision_kit.RESTORED_CACHE_HOOK = hook
    try:
        s = run(turn)
    finally:
        vision_kit.RESTORED_CACHE_HOOK = None
    c = s.context_check
    print(f"{label}: prompt={s.prompt_tokens} restored={c['restored_tokens']} ({c['restore_source']}) "
          f"scored={c['scored_tokens']} mean={c.get('surprise_mean')} p90={c.get('surprise_p90')} ms={c['ms']}",
          flush=True)


def prepare_wrong_context():
    other_history, other_turn = unrelated(uuid.uuid4().hex)
    run(other_history)
    vision_kit.RESTORED_CACHE_HOOK = capture
    try:
        run(other_turn)
    finally:
        vision_kit.RESTORED_CACHE_HOOK = None


for repeat in range(2):
    trial(f"intact#{repeat}")
    trial(f"C1-zero-recurrent#{repeat}", zero_recurrent)
    trial(f"C2-shuffle-kv+zero-recurrent#{repeat}", shuffle_kv)
    trial(f"C3-zero-all#{repeat}", zero_all)
    trial(f"C4-wrong-conversation#{repeat}", swap_in, prepare_wrong_context)
