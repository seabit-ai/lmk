"""exp07: DFlash 2 drafter via mlx-vlm 0.6.16's own generate_step (B=1), against plain decoding.
usage: run.py <out_dir>      (run inside the scratch venv that has mlx-vlm==0.6.16; needs no lmk code)
Arms: none · dflash2 (the drafter's own block size) · dflash2 at temperature 1.0 (acceptance only).
"""
import glob, hashlib, json, os, sys, time
from pathlib import Path
import mlx.core as mx
import mlx_vlm
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.speculative.common import speculative_stats_snapshot, speculative_stats_since

MAIN = "lmstudio-community/Qwen3.8-27B-MLX-4bit"
DRAFT = "incoai/Qwen3.8-27B-DFlash2"
EOS = {248046, 248044}
MAX_TOKENS = 400
REPS = 2


def snapshot(repo):
    paths = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--" + repo.replace("/", "--") + "/snapshots/*")),
                   key=os.path.getmtime)
    assert paths, repo
    return Path(paths[-1])


# exp01's run.py imports mlx_engine at module level, which this venv does not have: read its
# PROMPTS literal out of the source instead of importing the module.
import ast
_exp01 = (Path(__file__).resolve().parent.parent / "exp01-27b-draft-gain" / "run.py").read_text()
_ns = {}
for _node in ast.parse(_exp01).body:               # every top-level assignment, in order (PROMPTS builds on PARA etc.)
    if isinstance(_node, ast.Assign):
        exec(ast.get_source_segment(_exp01, _node), _ns)
PROMPTS = _ns["PROMPTS"]


def run_one(model, tok, toks, draft, kind, temperature=0.0, block=None):
    kwargs = dict(max_tokens=MAX_TOKENS, temperature=temperature, prefill_step_size=512)
    if draft is not None:
        kwargs.update(draft_model=draft, draft_kind=kind)
        if block:
            kwargs["draft_block_size"] = block
        snap = speculative_stats_snapshot(draft)
    ids = mx.array([toks], dtype=mx.int32)
    out = []; t0 = time.perf_counter(); first = None
    for token, _ in generate_step(ids, model, None, None, **kwargs):
        t = int(token.item()) if hasattr(token, "item") else int(token)
        if first is None:
            first = time.perf_counter()
        if t in EOS:
            break
        out.append(t)
    end = time.perf_counter()
    text = tok.decode(out)
    r = dict(prompt_tokens=len(toks), completion_tokens=len(out), first_token_ms=(first - t0) * 1000,
             total_ms=(end - t0) * 1000, decode_tok_s=(len(out) - 1) / max(1e-6, end - first),
             text=text, sha=hashlib.sha1(text.encode()).hexdigest()[:10])
    if draft is not None:
        rounds, accepted, drafted = speculative_stats_since(draft, snap)
        r.update(rounds=rounds, accepted=accepted, drafted=drafted, tokens_per_round=(len(out) / rounds) if rounds else None)
    mx.clear_cache()
    return r


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    path = snapshot(MAIN)
    t = time.perf_counter()
    model = mlx_vlm.utils.load_model(path, lazy=False, trust_remote_code=False)
    proc = mlx_vlm.utils.load_processor(path)
    tok = getattr(proc, "tokenizer", proc)
    print(f"main loaded in {time.perf_counter()-t:.1f}s (mlx_vlm {mlx_vlm.__version__})", flush=True)
    prompts = {}
    for k, v in PROMPTS.items():
        s = tok.apply_chat_template([{"role": "user", "content": v}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        prompts[k] = tok.encode(s, add_special_tokens=False)
    run_one(model, tok, prompts["story"], None, None)  # warm-up
    base = {}
    for name, toks in prompts.items():
        for rep in range(1, REPS + 1):
            r = run_one(model, tok, toks, None, None); r.update(arm="none", prompt=name, rep=rep)
            (out / f"none-{name}-run{rep}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
            base[name] = r
            print(f"none    {name:9} run{rep}  decode {r['decode_tok_s']:6.1f} tok/s  tokens {r['completion_tokens']:3}  ttft {r['first_token_ms']:5.0f} ms  sha {r['sha']}", flush=True)
    t = time.perf_counter(); draft, kind = load_drafter(str(snapshot(DRAFT)))
    print(f"draft loaded in {time.perf_counter()-t:.1f}s kind={kind} block={getattr(getattr(draft, 'config', None), 'dflash_config', None)}", flush=True)
    run_one(model, tok, prompts["story"], draft, kind)  # warm-up
    for name, toks in prompts.items():
        for rep in range(1, REPS + 1):
            r = run_one(model, tok, toks, draft, kind); r.update(arm="dflash2", prompt=name, rep=rep)
            (out / f"dflash2-{name}-run{rep}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
            same = "identical" if r["sha"] == base[name]["sha"] else "DIFFERENT"
            print(f"dflash2 {name:9} run{rep}  decode {r['decode_tok_s']:6.1f} tok/s  tokens {r['completion_tokens']:3}  ttft {r['first_token_ms']:5.0f} ms  sha {r['sha']}  {same}  "
                  f"x{r['decode_tok_s']/base[name]['decode_tok_s']:.2f}  tok/round {r['tokens_per_round']:.2f}  accepted {r['accepted']}/{r['drafted']}", flush=True)
    for name in ("code", "story"):
        r = run_one(model, tok, prompts[name], draft, kind, temperature=1.0); r.update(arm="dflash2-temp1", prompt=name, rep=1)
        (out / f"dflash2-temp1-{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        print(f"dflash2@temp1 {name:9}  decode {r['decode_tok_s']:6.1f} tok/s  tokens {r['completion_tokens']:3}  tok/round {r['tokens_per_round']:.2f}  accepted {r['accepted']}/{r['drafted']} ({r['accepted']/max(1,r['drafted']):.0%})", flush=True)


def run_one_block(model, tok, toks, draft, kind, block):
    return run_one(model, tok, toks, draft, kind, block=block)


if __name__ == "__main__":
    main()
