"""exp15: two requests drafting at once (MAX_ROUND_ROWS = 2) on today's verify path (plain forward + recorded rollback).
usage: smoke.py <out_dir>    env: KV_BITS=8, ROWS=1|2 (1 = today's behaviour, the control)
Each pair runs its two prompts concurrently; every output is compared with the same prompt decoded alone, plain.
"""
import glob, hashlib, json, os, sys, threading, time
from pathlib import Path

from mlx_engine.generate import create_generator, load_draft_model, load_model
from mlx_engine.model_kit.batched_vision import speculative

sys.path.insert(0, os.path.expanduser("~/src/lmk/research/2026-09-23-speculative-decoding/exp01-27b-draft-gain"))
from run import PROMPTS  # story / code / copyedit

MAIN = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--lmstudio-community--Qwen3.8-27B-MLX-4bit/snapshots/*")))[-1]
DRAFT = os.path.expanduser("~/.cache/lmk-research/qwen3.8-27b-mtp-draft")
MAX_TOKENS = 400
KV_BITS = int(os.environ.get("KV_BITS") or 0) or None
ROWS = int(os.environ.get("ROWS") or 2)
PAIRS = [("code", "story"), ("code", "code"), ("code", "copyedit"), ("story", "copyedit")]


def tokens_for(kit, text):
    s = kit.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return kit.tokenizer.encode(s, add_special_tokens=False)


def run(kit, toks, request_id, spec=None):
    t0 = time.perf_counter(); first = None; n = 0; text = ""
    for r in create_generator(kit, list(toks), max_tokens=MAX_TOKENS, temp=0.0, request_id=request_id, speculative_decoding_toggle=spec):
        if r.tokens and first is None: first = time.perf_counter()
        n += len(r.tokens); text += r.text
        if r.stop_condition: break
    end = time.perf_counter()
    return dict(tokens=n, text=text, sha=hashlib.sha1(text.encode()).hexdigest()[:10], seconds=end - t0,
                decode_tok_s=round((n - 1) / max(1e-6, end - first), 1))


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    speculative.SpeculativeGenerationBatch.MAX_ROUND_ROWS = ROWS
    kit = load_model(Path(MAIN), max_kv_size=32768, max_seq_nums=2, kv_bits=KV_BITS)
    prompts = {k: tokens_for(kit, v) for k, v in PROMPTS.items()}
    run(kit, prompts["story"], "warm")
    base = {k: run(kit, t, f"base-{k}") for k, t in prompts.items()}
    for k, r in base.items(): print(f"plain  {k:9} {r['decode_tok_s']:6.1f} tok/s  sha {r['sha']}", flush=True)
    load_draft_model(kit, DRAFT)
    run(kit, prompts["story"], "warm-spec")
    alone = {k: run(kit, t, f"alone-{k}") for k, t in prompts.items()}
    for k, r in alone.items(): print(f"alone  {k:9} {r['decode_tok_s']:6.1f} tok/s  sha {r['sha']}  {'identical' if r['sha']==base[k]['sha'] else 'DIFFERENT'}", flush=True)
    results = {"rows": ROWS, "kv_bits": KV_BITS, "base": base, "alone": alone, "pairs": []}
    for a, b in PAIRS:
        got = {}
        def worker(slot, name):
            got[slot] = run(kit, prompts[name], f"pair-{a}-{b}-{slot}")
        r0 = kit._drafter.rounds if kit._drafter else 0
        t0 = time.perf_counter()
        ths = [threading.Thread(target=worker, args=(0, a)), threading.Thread(target=worker, args=(1, b))]
        [t.start() for t in ths]; [t.join() for t in ths]
        wall = time.perf_counter() - t0
        rounds = (kit._drafter.rounds if kit._drafter else 0) - r0
        tot = got[0]["tokens"] + got[1]["tokens"]
        verdict = [("identical" if got[i]["sha"] == base[n]["sha"] else "DIFFERENT") for i, n in enumerate((a, b))]
        print(f"pair   {a}+{b:9} {got[0]['decode_tok_s']:5.1f} + {got[1]['decode_tok_s']:5.1f} tok/s  total {tot/wall:5.1f} tok/s  "
              f"{a}:{verdict[0]} {b}:{verdict[1]}  spec rounds {rounds}", flush=True)
        results["pairs"].append(dict(a=a, b=b, each=got, wall=wall, total_tok_s=tot / wall, verdict=verdict, rounds=rounds))
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
