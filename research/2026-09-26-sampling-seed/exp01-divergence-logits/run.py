"""exp01: logits at the first diverging position. See README.md.
PYTHONPATH=.engine/mlx-engine:. .venv/bin/python research/2026-09-26-sampling-seed/exp01-divergence-logits/run.py cold-vs-restore|spec"""
import sys
import uuid

sys.path.insert(0, "tests")
from itest_model import model_dir  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_engine.utils import sampling as eng_sampling  # noqa: E402
from lmk.engine import MlxEngine  # noqa: E402

MODE = sys.argv[1]
DRAFT = "/Users/xinkai/.cache/huggingface/hub/models--seabit-ai--Qwen3.8-27B-DFlash2-4bit/snapshots/local-quant-2026-09-24"
TOOLS = [{"type": "function", "function": {
    "name": "file_read", "description": "Read a text file and return its contents.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}},
                   "required": ["path"]}}}]

record = {}
orig = eng_sampling.SeededSampler.sample_target


def traced(self, logprobs, *, row_ids, positions):
    drawn = orig(self, logprobs, row_ids=row_ids, positions=positions)
    lp = logprobs if logprobs.ndim == 2 else logprobs[None]
    top = mx.argsort(-lp, axis=-1)[:, :20]
    vals = mx.take_along_axis(lp, top, axis=-1)
    for i, p in enumerate(positions):
        record[int(p)] = (top[i].tolist(), vals[i].tolist(), int(drawn.reshape(-1)[i].item()), i, len(positions))
    return drawn


eng_sampling.SeededSampler.sample_target = traced
engine = MlxEngine("exp", model_dir(), 32768, draft_path=DRAFT if MODE.startswith("spec") else None)
fmt = engine.chat_format()
tok = engine._kit.tokenizer


def run(prompt, seed, speculative=None):
    record.clear()
    sampling = {"temp": 1.0, "top_p": 0.95, "top_k": 20, "seed": seed}
    if speculative is not None:
        sampling["speculative_decoding_toggle"] = speculative
    g = engine.generate(prompt, max_tokens=160, request_id=f"exp-{uuid.uuid4().hex[:8]}", on_prefill=lambda *a: True,
                        sampling=sampling)
    text = "".join(g)
    return dict(record), g.stats, text


def profile(label, a, b):
    """per position: total variation over the union of both top-20s, and max |dlogprob| among tokens either side
    gives p > 0.05; grouped by where the spec run (b) drew the position: offset in its verify block"""
    import math
    groups = {}
    for p in sorted(set(a) & set(b)):
        if a[p][2] != b[p][2]:
            break
        pa, pb = dict(zip(a[p][0], a[p][1])), dict(zip(b[p][0], b[p][1]))
        tv = 0.5 * sum(abs(math.exp(pa.get(t, -30)) - math.exp(pb.get(t, -30))) for t in set(pa) | set(pb))
        big = [abs(pa[t] - pb[t]) for t in set(pa) & set(pb) if max(pa[t], pb[t]) > math.log(0.05)]
        key = "single" if b[p][4] == 1 else f"block offset {min(b[p][3], 3)}{'+' if b[p][3] >= 3 else ''}"
        g = groups.setdefault(key, [0, 0.0, 0.0])
        g[0] += 1
        g[1] = max(g[1], tv)
        g[2] = max(g[2], max(big) if big else 0.0)
    for key, (n, tv, big) in sorted(groups.items()):
        print(f"   {label} [{key}] n={n} max TV={tv:.4f} max|dlogprob| (p>0.05)={big:.4f}", flush=True)


def compare(label, a, b):
    positions = sorted(set(a) & set(b))
    j = next((p for p in positions if a[p][2] != b[p][2]), None)
    worst = 0.0
    for p in positions:
        if j is not None and p > j:
            break
        ta, va = a[p][0], a[p][1]
        tb, vb = b[p][0], b[p][1]
        da, db = dict(zip(ta, va)), dict(zip(tb, vb))
        for t in set(da) & set(db):
            if p < (j if j is not None else 10**9) or p == j:
                worst = max(worst, abs(da[t] - db[t]))
    if j is None:
        print(f"{label}: identical over {len(positions)} positions, max|dlogprob| (top-20, shared) {worst:.4f}", flush=True)
        return
    ta, va, xa = a[j][:3]
    tb, vb, xb = b[j][:3]
    da, db = dict(zip(ta, va)), dict(zip(tb, vb))
    before = 0.0
    for p in positions:
        if p >= j:
            break
        pa, pb = dict(zip(a[p][0], a[p][1])), dict(zip(b[p][0], b[p][1]))
        before = max([before] + [abs(pa[t] - pb[t]) for t in set(pa) & set(pb)])
    at_j = max(abs(da[t] - db[t]) for t in set(da) & set(db))
    show = lambda t: repr(tok.decode([t]))  # noqa: E731
    print(f"{label}: first divergence at generated position {j}; max|dlogprob| before it {before:.4f}, at it {at_j:.4f}",
          flush=True)
    print(f"   A drew {show(xa)} p={2.718281828 ** da.get(xa, -99):.3f} (B gives it {2.718281828 ** db.get(xa, -99):.3f}); "
          f"B drew {show(xb)} p={2.718281828 ** db.get(xb, -99):.3f} (A gives it {2.718281828 ** da.get(xb, -99):.3f})", flush=True)
    print("   A top5:", [(show(t), round(v, 3)) for t, v in zip(ta[:5], va[:5])], flush=True)
    print("   B top5:", [(show(t), round(v, 3)) for t, v in zip(tb[:5], vb[:5])], flush=True)


def prompt_for(tools):
    messages = [{"role": "system", "content": f"Session {uuid.uuid4().hex}. You are a creative assistant."},
                {"role": "user", "content": "Invent a name for a new colour and describe it in two sentences."}]
    return fmt.render(messages, TOOLS if tools else None)


if MODE in ("cold-vs-restore", "spec-cold-vs-restore"):
    for tools in ((True,) if MODE.startswith("spec") else (False, True)):
        for seed in (11, 12, 13):
            prompt = prompt_for(tools)
            cold, s1, _ = run(prompt, seed)
            restored, s2, _ = run(prompt, seed)
            profile(f"tools={tools} seed={seed}", cold, restored)
            if MODE.startswith("spec"):
                for p in range(4):
                    if p in cold and p in restored:
                        print(f"   pos {p}: cold drawn by a call of {cold[p][4]} (offset {cold[p][3]}), restored by a call of "
                              f"{restored[p][4]} (offset {restored[p][3]}); tokens {cold[p][2]} / {restored[p][2]}", flush=True)
            compare(f"tools={tools} seed={seed} cold(cached {s1.cached_tokens}) vs restored(cached {s2.cached_tokens})",
                    cold, restored)
            again, s3, _ = run(prompt, seed)
            compare(f"tools={tools} seed={seed} restored vs restored again", restored, again)
else:
    for seed in (21, 22, 23):
        prompt = fmt.render([{"role": "system", "content": f"Session {uuid.uuid4().hex}."},
                             {"role": "user", "content": "Write a four-line poem about a lighthouse."}], None)
        run(prompt, seed, speculative=False)        # reads the prompt; both runs below restore the same prefix
        plain, s1, _ = run(prompt, seed, speculative=False)
        spec, s2, _ = run(prompt, seed, speculative=True)
        profile(f"seed={seed}", plain, spec)
        compare(f"seed={seed} plain vs spec (drafted {s2.draft_drafted} accepted {s2.draft_accepted}; cached "
                f"{s1.cached_tokens}/{s2.cached_tokens})", plain, spec)
        plain2, _, _ = run(prompt, seed, speculative=False)
        compare(f"seed={seed} plain vs plain again", plain, plain2)
