"""exp01: where the context check's time goes, and why a restored prefill scored nothing.
Run from ~/src/lmk-seed: PYTHONPATH=.engine/mlx-engine:. .venv/bin/python research/2026-09-26-context-check/exp01-cost-and-restore-path/run.py"""
import sys
import time
import uuid

sys.path.insert(0, "tests")
from itest_model import model_dir  # noqa: E402

import mlx_engine.model_kit.batched_vision.context_check as cc  # noqa: E402
from lmk.engine import MlxEngine  # noqa: E402

orig_add = cc.SurpriseMeter.add


def traced_add(self, start, hidden):
    print(f"  meter.add start={start} hidden={None if hidden is None else tuple(hidden.shape)} lo/hi={self._lo}/{self._hi}",
          flush=True)
    return orig_add(self, start, hidden)


orig_wants = cc.SurpriseMeter.wants


def traced_wants(self, start, n):
    w = orig_wants(self, start, n)
    print(f"  meter.wants start={start} n={n} -> {w}", flush=True)
    return w


cc.SurpriseMeter.add = traced_add
cc.SurpriseMeter.wants = traced_wants

engine = MlxEngine("exp", model_dir(), 32768)
fmt = engine.chat_format()
names = [f"{w}-{i * 7919 % 1000:03d}" for i, w in enumerate(
    "amber basalt cobalt dune ember fjord garnet harbor indigo jasper kelp lumen marble nectar onyx pewter "
    "quartz russet sable tundra umber velvet willow xenon yarrow zephyr".split())]
history = [{"role": "system", "content": f"Session {uuid.uuid4().hex}. You are kitten, a coding agent."},
           {"role": "user", "content": "Here are the release codenames, in order: " + ", ".join(names) + ". " +
            " ".join(f"Note {i}: codename {n} ships in week {i + 3}." for i, n in enumerate(names))},
           {"role": "assistant", "content": "Understood."}]
turn = history + [{"role": "user", "content": "Request A. " + "; ".join(f"{n}: week {i + 3}" for i, n in enumerate(names))}]


def run(name, messages):
    t0 = time.perf_counter()
    first = {}

    def on_prefill(p, total, cached):
        return True

    g = engine.generate(fmt.render(messages, None), max_tokens=1, request_id=f"exp-{uuid.uuid4().hex[:8]}",
                        on_prefill=on_prefill, sampling={"temp": 0.0})
    for _ in g:
        first.setdefault("ms", (time.perf_counter() - t0) * 1000)
    wall = (time.perf_counter() - t0) * 1000
    c = g.stats.context_check
    print(f"{name}: prompt={g.stats.prompt_tokens} cached={g.stats.cached_tokens} wall_ms={wall:.0f} check={c}", flush=True)


run("cold", history)
run("restored", turn)

# projection alone, 512 rows of the model's own hidden size, both ways the check could call it
import mlx.core as mx  # noqa: E402

lm = engine._kit.model.language_model
h = mx.random.normal((1, 512, lm.args.hidden_size)).astype(mx.bfloat16)
mx.eval(h)
for label, fn in (("speculative_logits_from_hidden", lm.speculative_logits_from_hidden), ("lm_head", lm.lm_head)):
    for _ in range(2):
        t = time.perf_counter()
        lg = fn(h).reshape(512, -1).astype(mx.float32)
        lp = mx.take_along_axis(lg, mx.zeros((512, 1), dtype=mx.int32), axis=-1)[:, 0] - mx.logsumexp(lg, axis=-1)
        mx.eval(lp)
        print(f"projection {label}: {(time.perf_counter() - t) * 1000:.1f} ms for 512 rows", flush=True)
