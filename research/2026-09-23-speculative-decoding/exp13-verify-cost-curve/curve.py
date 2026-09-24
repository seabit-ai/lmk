"""exp13 A: one forward of T tokens against a prefilled cache, timed as a dependent chain.

usage: curve.py <out_dir>
"""
import glob, json, os, statistics, sys, time
from pathlib import Path

import mlx.core as mx
import mlx_vlm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_qmm

MAIN = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--lmstudio-community--Qwen3.8-27B-MLX-4bit/snapshots/*")))[-1]
CONTEXTS = [4096, 32768]
WIDTHS = {"stock": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16], "fast": [5, 6, 7, 8], "fastwide": [8, 9, 10, 12, 16]}
CHECK_WIDTHS = [6, 8, 12, 16]
WARMUP, TIMED = 3, 15
PREFILL_CHUNK = 2048


def set_arm(arm):
    fast_qmm.disable()
    os.environ.pop("MLXLM_FAST_QMM_WIDE", None)
    if arm in ("fast", "fastwide"):
        fast_qmm.enable()
    if arm == "fastwide":
        os.environ["MLXLM_FAST_QMM_WIDE"] = "1"


def filler_tokens(tok, n):
    text = (Path(__file__).resolve().parents[3] / "README.md").read_text()
    ids = tok.encode(text, add_special_tokens=False)
    return (ids * (n // len(ids) + 1))[:n]


class Snapshot:
    """KV layers roll back by trimming; linear-attention layers get their state arrays back."""
    def __init__(self, cache):
        self.cache = cache
        self.offsets = {i: c.offset for i, c in enumerate(cache) if hasattr(c, "offset")}
        self.states = {i: list(c.cache) for i, c in enumerate(cache) if not hasattr(c, "offset")}

    def restore(self):
        for i, off in self.offsets.items():
            self.cache[i].trim(self.cache[i].offset - off)
        for i, st in self.states.items():
            self.cache[i].cache = list(st)


def forward(lm, cache, token, width):
    out = lm(mx.full((1, width), token, dtype=mx.int32), cache=cache)
    return out.logits


def time_width(lm, cache, snap, token, width):
    samples = []
    for i in range(WARMUP + TIMED):
        t0 = time.perf_counter()
        logits = forward(lm, cache, token, width)
        nxt = mx.argmax(logits[0, -1])
        mx.eval(nxt)
        dt = (time.perf_counter() - t0) * 1000
        snap.restore()
        token = int(nxt.item())   # the next input depends on this output: no queued overlap
        if i >= WARMUP:
            samples.append(dt)
    samples.sort()
    return dict(median_ms=round(statistics.median(samples), 2), p10_ms=round(samples[1], 2),
                p90_ms=round(samples[-2], 2)), token


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    model = mlx_vlm.utils.load_model(Path(MAIN), lazy=False, trust_remote_code=False)
    tok = getattr(mlx_vlm.utils.load_processor(Path(MAIN)), "tokenizer", None)
    lm = model.language_model
    results = dict(model=MAIN, mlx=mx.__version__, mlx_vlm=mlx_vlm.__version__, curves={}, check={})
    for ctx in CONTEXTS:
        set_arm("stock")
        cache = lm.make_cache()
        ids = filler_tokens(tok, ctx)
        t0 = time.perf_counter()
        for s in range(0, ctx, PREFILL_CHUNK):
            lm(mx.array([ids[s:s + PREFILL_CHUNK]], dtype=mx.int32), cache=cache)
            mx.eval([x for c in cache for x in (c.state if hasattr(c, "offset") else c.cache) if x is not None])
            mx.clear_cache()
        print(f"ctx {ctx}: prefill {time.perf_counter() - t0:.1f}s", flush=True)
        snap = Snapshot(cache)
        token = ids[-1]
        for arm, widths in WIDTHS.items():
            set_arm(arm)
            for w in widths:
                r, token = time_width(lm, cache, snap, token, w)
                results["curves"].setdefault(str(ctx), {}).setdefault(arm, {})[str(w)] = r
                print(f"ctx {ctx:6} {arm:8} T={w:2}  median {r['median_ms']:7.2f} ms  p10 {r['p10_ms']:7.2f}  p90 {r['p90_ms']:7.2f}", flush=True)
        for w in CHECK_WIDTHS:
            set_arm("stock"); a = forward(lm, cache, token, w).astype(mx.float32); mx.eval(a); snap.restore()
            set_arm("fastwide"); b = forward(lm, cache, token, w).astype(mx.float32); mx.eval(b); snap.restore()
            diff = float(mx.max(mx.abs(a - b)).item())
            same = int(mx.sum(mx.argmax(a[0], axis=-1) == mx.argmax(b[0], axis=-1)).item())
            results["check"].setdefault(str(ctx), {})[str(w)] = dict(max_abs_logit_diff=round(diff, 4), argmax_same=f"{same}/{w}")
            print(f"ctx {ctx:6} check T={w:2}  max |logit diff| {diff:.4f}  argmax same {same}/{w}", flush=True)
        set_arm("stock")
        del cache, snap
        mx.clear_cache()
    (out / "curve.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
