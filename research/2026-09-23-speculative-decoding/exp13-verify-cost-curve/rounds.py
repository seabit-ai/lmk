"""exp13 B: the engine's speculative round — where the milliseconds go, and what the small-M kernel buys end to end.

usage: rounds.py <out_dir>   (needs PYTHONPATH=<engine worktree>:<lmk>)
"""
import glob, hashlib, json, os, statistics, sys, time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx_engine.model_kit.batched_vision.speculative as spec
from mlx_engine.generate import create_generator, load_draft_model, load_model, unload_draft_model

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "exp01-27b-draft-gain"))
import fast_qmm
from run import PROMPTS  # story / code / copyedit

HUB = os.path.expanduser("~/.cache/huggingface/hub")
MAIN = sorted(glob.glob(f"{HUB}/models--lmstudio-community--Qwen3.8-27B-MLX-4bit/snapshots/*"))[-1]
DRAFTERS = {
    "mtp": sorted(glob.glob(f"{HUB}/models--seabit-ai--Qwen3.8-27B-MTP-draft/snapshots/*"))[-1],
    "dflash2": f"{HUB}/models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5",
}
ARMS = ["stock", "fastwide"]
MAX_TOKENS = 400


def set_arm(arm):
    fast_qmm.disable()
    os.environ.pop("MLXLM_FAST_QMM_WIDE", None)
    if arm == "fastwide":
        fast_qmm.enable()
        os.environ["MLXLM_FAST_QMM_WIDE"] = "1"


def tokens_for(kit, text):
    s = kit.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                          tokenize=False, enable_thinking=False)
    return kit.tokenizer.encode(s, add_special_tokens=False)


def run(kit, toks, request_id):
    t0 = time.perf_counter(); first = None; n = 0; text = ""
    for r in create_generator(kit, list(toks), max_tokens=MAX_TOKENS, temp=0.0, request_id=request_id):
        if r.tokens and first is None: first = time.perf_counter()
        n += len(r.tokens); text += r.text
        if r.stop_condition: break
    end = time.perf_counter()
    return dict(tokens=n, sha=hashlib.sha1(text.encode()).hexdigest()[:10],
                decode_tok_s=round((n - 1) / max(1e-6, end - first), 1))


# --- instrumentation: each part synchronised on both sides, its own outputs evaluated ---
PARTS = defaultdict(list)
WIDTHS = []


def _arrays(obj):
    if isinstance(obj, mx.array):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [a for o in obj for a in _arrays(o)]
    return []


def _cache_state(cache):
    return [a for c in cache for a in _arrays(getattr(c, "cache", None))]


def timed(name, fn, outputs):
    def wrapper(*args, **kwargs):
        mx.synchronize()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        mx.eval(outputs(result, args, kwargs))
        mx.synchronize()
        PARTS[name].append((time.perf_counter() - t0) * 1000)
        return result
    return wrapper


def instrument(kit):
    lm = getattr(kit.model, "language_model", kit.model)
    saved = dict(verify=spec._verify_block, mtp_draft=spec._mtp._mtp_draft_block_active,
                 sround=spec.speculative_round, dround=spec.dflash_round, rollback=lm.rollback_speculative_cache)

    def verify_outputs(r, args, kwargs):
        WIDTHS.append(int(args[1].shape[1]))
        return [a for a in (r.hidden, r.target_tokens, r.logits) if a is not None]

    spec._verify_block = timed("verify", saved["verify"], verify_outputs)
    spec._mtp._mtp_draft_block_active = timed("draft", saved["mtp_draft"], lambda r, a, k: _arrays(r))
    spec.speculative_round = timed("round", saved["sround"], lambda r, a, k: [r.hidden])
    spec.dflash_round = timed("round", saved["dround"], lambda r, a, k: [r.hidden])
    lm.rollback_speculative_cache = timed("rollback", saved["rollback"], lambda r, a, k: _cache_state(a[0]))
    drafter_model = kit._drafter.model
    if hasattr(drafter_model, "draft_block"):
        saved["dflash_draft"] = drafter_model.draft_block
        drafter_model.draft_block = timed("draft", saved["dflash_draft"], lambda r, a, k: _arrays(r))

    def restore():
        spec._verify_block = saved["verify"]
        spec._mtp._mtp_draft_block_active = saved["mtp_draft"]
        spec.speculative_round = saved["sround"]
        spec.dflash_round = saved["dround"]
        lm.rollback_speculative_cache = saved["rollback"]
        if "dflash_draft" in saved:
            drafter_model.draft_block = saved["dflash_draft"]
    return restore


def breakdown():
    rounds = len(PARTS["round"])
    per_round = {k: round(sum(v) / rounds, 2) for k, v in PARTS.items() if k != "round"}
    per_round["round"] = round(statistics.mean(PARTS["round"]), 2)
    per_round["other"] = round(per_round["round"] - sum(v for k, v in per_round.items() if k != "round"), 2)
    hist = {w: WIDTHS.count(w) for w in sorted(set(WIDTHS))}
    return dict(rounds=rounds, per_round_ms=per_round, rollback_rounds=len(PARTS["rollback"]),
                verify_median_ms_by_width={w: round(statistics.median(
                    [t for t, ww in zip(PARTS["verify"], WIDTHS) if ww == w]), 2) for w in hist},
                verify_width_histogram=hist)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    kit = load_model(Path(MAIN), max_kv_size=32768, max_seq_nums=2)
    prompts = {k: tokens_for(kit, v) for k, v in PROMPTS.items()}
    results = dict(model=MAIN, plain={}, spec={}, breakdown={})
    run(kit, prompts["story"], "warm")
    for arm in ARMS:
        set_arm(arm)
        for k, t in prompts.items():
            r = results["plain"].setdefault(arm, {})[k] = run(kit, t, f"plain-{arm}-{k}")
            print(f"plain   {arm:8} {k:9} {r['decode_tok_s']:6.1f} tok/s  tokens {r['tokens']:3}  sha {r['sha']}", flush=True)
    for name, path in DRAFTERS.items():
        set_arm("stock")
        load_draft_model(kit, path)
        run(kit, prompts["story"], f"warm-{name}")
        for arm in ARMS:
            set_arm(arm)
            for k, t in prompts.items():
                r0 = (kit._drafter.rounds, kit._drafter.accepted)
                r = run(kit, t, f"{name}-{arm}-{k}")
                r.update(rounds=kit._drafter.rounds - r0[0], accepted=kit._drafter.accepted - r0[1])
                results["spec"].setdefault(name, {}).setdefault(arm, {})[k] = r
                base = results["plain"]["stock"][k]
                same = "identical" if r["sha"] == base["sha"] else "DIFFERENT"
                print(f"{name:7} {arm:8} {k:9} {r['decode_tok_s']:6.1f} tok/s  x{r['decode_tok_s'] / base['decode_tok_s']:.2f}  "
                      f"rounds {r['rounds']:3}  tok/round {r['tokens'] / max(1, r['rounds']):.2f}  sha {r['sha']} {same}", flush=True)
            PARTS.clear(); WIDTHS.clear()
            restore = instrument(kit)
            run(kit, prompts["code"], f"{name}-{arm}-code-instrumented")
            restore()
            b = results["breakdown"].setdefault(name, {})[arm] = breakdown()
            print(f"{name:7} {arm:8} code breakdown  {json.dumps(b)}", flush=True)
        unload_draft_model(kit)
    set_arm("stock")
    (out / "rounds.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
