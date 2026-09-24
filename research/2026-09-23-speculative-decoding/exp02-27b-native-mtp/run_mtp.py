"""exp02: native Qwen3.8 MTP drafter via mlx-vlm generate_step (B=1).

usage: run_mtp.py <out_dir> [arms...]   arms default: none mtp-b2 mtp-b3 mtp-b4 mtp-b6
"""
import json, sys, time, glob, os, hashlib
from pathlib import Path
import mlx.core as mx
import mlx_vlm
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.speculative.common import speculative_stats_snapshot, speculative_stats_since

MAIN = "lmstudio-community/Qwen3.8-27B-MLX-4bit"
DRAFT = os.path.expanduser("~/.cache/lmk-research/qwen3.8-27b-mtp-draft")
EOS = {248046, 248044}
MAX_TOKENS = 400
REPS = 2

def snapshot(repo):
    hub = os.path.expanduser("~/.cache/huggingface/hub/models--" + repo.replace("/", "--") + "/snapshots/*")
    paths = sorted(glob.glob(hub), key=os.path.getmtime); assert paths, repo
    return Path(paths[-1])

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp01-27b-draft-gain"))
from run import PROMPTS  # same three prompts

def run_one(model, tok, toks, draft, block):
    kwargs = dict(max_tokens=MAX_TOKENS, temperature=0.0, prefill_step_size=512)
    if draft is not None:
        kwargs.update(draft_model=draft, draft_kind="mtp", draft_block_size=block)
        snap = speculative_stats_snapshot(draft)
    ids = mx.array([toks], dtype=mx.int32)
    out = []; t0 = time.perf_counter(); first = None
    for token, _ in generate_step(ids, model, None, None, **kwargs):
        t = int(token.item()) if hasattr(token, "item") else int(token)
        if first is None: first = time.perf_counter()
        if t in EOS: break
        out.append(t)
    end = time.perf_counter()
    r = dict(prompt_tokens=len(toks), completion_tokens=len(out), first_token_ms=(first - t0) * 1000,
             total_ms=(end - t0) * 1000, decode_tok_s=(len(out) - 1) / max(1e-6, end - first),
             text=tok.decode(out), sha=hashlib.sha1(tok.decode(out).encode()).hexdigest()[:10])
    if draft is not None:
        rounds, accepted, drafted = speculative_stats_since(draft, snap)
        r.update(rounds=rounds, accepted=accepted, drafted=drafted,
                 tokens_per_round=(len(out) / rounds) if rounds else None)
    mx.clear_cache()
    return r

def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    arms = sys.argv[2:] or ["none", "mtp-b2", "mtp-b3", "mtp-b4", "mtp-b6"]
    path = snapshot(MAIN)
    t = time.perf_counter()
    model = mlx_vlm.utils.load_model(path, lazy=False, trust_remote_code=False)
    proc = mlx_vlm.utils.load_processor(path)
    tok = getattr(proc, "tokenizer", proc)
    print(f"main loaded in {time.perf_counter()-t:.1f}s", flush=True)
    prompts = {}
    for k, v in PROMPTS.items():
        s = tok.apply_chat_template([{"role": "user", "content": v}], add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)
        prompts[k] = tok.encode(s, add_special_tokens=False)
    draft = None
    run_one(model, tok, prompts["story"], None, None)  # warm-up
    for arm in arms:
        if arm == "none":
            d, block = None, None
        else:
            if draft is None:
                t = time.perf_counter(); draft, kind = load_drafter(DRAFT)
                print(f"draft loaded in {time.perf_counter()-t:.1f}s kind={kind}", flush=True)
                run_one(model, tok, prompts["story"], draft, 3)  # warm-up
            d, block = draft, int(arm.split("-b")[1])
        for name, toks in prompts.items():
            for rep in range(1, REPS + 1):
                r = run_one(model, tok, toks, d, block); r.update(arm=arm, prompt=name, rep=rep)
                (out / f"{arm}-{name}-run{rep}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
                extra = f"tok/round {r['tokens_per_round']:.2f}  accepted {r['accepted']}/{r['drafted']}" if d else ""
                print(f"{arm:7} {name:9} run{rep}  decode {r['decode_tok_s']:6.1f} tok/s  tokens {r['completion_tokens']:3}  "
                      f"ttft {r['first_token_ms']:5.0f} ms  sha {r['sha']}  {extra}", flush=True)

if __name__ == "__main__":
    main()
