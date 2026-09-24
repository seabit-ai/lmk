"""exp14 smoke: exp12 with the 4-bit DFlash 2 drafter (KV_BITS env: 8 to quantize the cache).

usage: smoke.py <out_dir>   (needs PYTHONPATH=<engine worktree>:<lmk>)
Runs each prompt greedy without and with the drafter (B=1), the sampled acceptance rate, then two prompts at once (B=2).
"""
import glob, hashlib, json, os, sys, threading, time
from pathlib import Path

from mlx_engine.generate import create_generator, load_draft_model, load_model

sys.path.insert(0, os.path.expanduser("~/src/lmk/research/2026-09-23-speculative-decoding/exp01-27b-draft-gain"))
from run import PROMPTS  # story / code / copyedit

MAIN = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--lmstudio-community--Qwen3.8-27B-MLX-4bit/snapshots/*")))[-1]
DRAFT = os.path.expanduser("~/.cache/lmk-research/qwen3.8-27b-dflash2-4bit")
MAX_TOKENS = 400
KV_BITS = int(os.environ.get("KV_BITS") or 0) or None


def tokens_for(kit, text):
    s = kit.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                          tokenize=False, enable_thinking=False)
    return kit.tokenizer.encode(s, add_special_tokens=False)


def run(kit, toks, request_id, spec=None, temp=0.0):
    t0 = time.perf_counter(); first = None; n = 0; text = ""
    for r in create_generator(kit, list(toks), max_tokens=MAX_TOKENS, temp=temp, request_id=request_id,
                              speculative_decoding_toggle=spec):
        if r.tokens and first is None: first = time.perf_counter()
        n += len(r.tokens); text += r.text
        if r.stop_condition: break
    end = time.perf_counter()
    return dict(tokens=n, text=text, sha=hashlib.sha1(text.encode()).hexdigest()[:10],
                ttft_ms=round((first - t0) * 1000), decode_tok_s=round((n - 1) / max(1e-6, end - first), 1))


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    kit = load_model(Path(MAIN), max_kv_size=32768, max_seq_nums=2, kv_bits=KV_BITS)
    prompts = {k: tokens_for(kit, v) for k, v in PROMPTS.items()}
    run(kit, prompts["story"], "warm")
    base = {k: run(kit, t, f"base-{k}") for k, t in prompts.items()}
    for k, r in base.items(): print(f"plain  {k:9} {r['decode_tok_s']:6.1f} tok/s  tokens {r['tokens']:3}  sha {r['sha']}", flush=True)
    load_draft_model(kit, DRAFT)
    run(kit, prompts["story"], "warm-spec")
    spec = {k: run(kit, t, f"spec-{k}") for k, t in prompts.items()}
    for k, r in spec.items():
        same = "identical" if r["sha"] == base[k]["sha"] else "DIFFERENT"
        print(f"draft  {k:9} {r['decode_tok_s']:6.1f} tok/s  tokens {r['tokens']:3}  sha {r['sha']}  {same}  x{r['decode_tok_s']/base[k]['decode_tok_s']:.2f}", flush=True)
    print("drafter stats rounds/accepted/drafted", kit._drafter.rounds, kit._drafter.accepted, kit._drafter.drafted, flush=True)
    r0 = (kit._drafter.rounds, kit._drafter.accepted, kit._drafter.drafted)
    s = run(kit, prompts["code"], "spec-sampled", temp=1.0)
    r1 = (kit._drafter.rounds, kit._drafter.accepted, kit._drafter.drafted)
    print(f"draft  code@temp1 {s['decode_tok_s']:6.1f} tok/s  tokens {s['tokens']}  accepted/drafted since {r1[1]-r0[1]}/{r1[2]-r0[2]} in {r1[0]-r0[0]} rounds", flush=True)
    results = {}
    def worker(name, toks):
        results[name] = run(kit, toks, f"pair-{name}")
    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(k, prompts[k])) for k in ("story", "code")]
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.perf_counter() - t0
    tot = sum(r["tokens"] for r in results.values())
    for k, r in results.items():
        print(f"pair   {k:9} {r['decode_tok_s']:6.1f} tok/s  tokens {r['tokens']}  sha {r['sha']}  {'identical' if r['sha']==base[k]['sha'] else 'DIFFERENT'}", flush=True)
    print(f"pair   total {tot} tokens in {wall:.1f}s = {tot/wall:.1f} tok/s aggregate", flush=True)
    (out / "results.json").write_text(json.dumps(dict(kv_bits=KV_BITS, base=base, spec=spec, sampled=s, pair=results, pair_wall_s=wall), indent=1, ensure_ascii=False))
    kit.shutdown() if hasattr(kit, "shutdown") else None

if __name__ == "__main__":
    main()
