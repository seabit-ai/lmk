"""exp07 extra arms: the same prompts with (a) the drafter quantized to 4 bits, (b) a fixed block size of 8.
usage: run_extra.py <out_dir> <quantized drafter dir or ->
"""
import json, sys, time
from pathlib import Path
import mlx_vlm
from mlx_vlm.speculative.drafters import load_drafter
sys.argv, argv = sys.argv[:1], sys.argv
sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -P leaves the script dir out
import run  # exp07's run.py: PROMPTS, run_one, snapshot, MAIN, DRAFT

out = Path(argv[1]); out.mkdir(parents=True, exist_ok=True)
qdir = None if argv[2] == "-" else argv[2]
model = mlx_vlm.utils.load_model(run.snapshot(run.MAIN), lazy=False, trust_remote_code=False)
tok = getattr(mlx_vlm.utils.load_processor(run.snapshot(run.MAIN)), "tokenizer", None)
prompts = {k: tok.encode(tok.apply_chat_template([{"role": "user", "content": v}], add_generation_prompt=True, tokenize=False,
                                                 enable_thinking=False), add_special_tokens=False) for k, v in run.PROMPTS.items()}
arms = [("dflash2-b8", str(run.snapshot(run.DRAFT)), 8)]
if qdir:
    arms.append(("dflash2-q4", qdir, None))
    arms.append(("dflash2-q4-b8", qdir, 8))
for arm, path, block in arms:
    draft, kind = load_drafter(path)
    run.run_one(model, tok, prompts["story"], draft, kind)  # warm-up
    for name, toks in prompts.items():
        kw = {}
        r = run.run_one_block(model, tok, toks, draft, kind, block) if block else run.run_one(model, tok, toks, draft, kind)
        r.update(arm=arm, prompt=name)
        (out / f"{arm}-{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        print(f"{arm:13} {name:9}  decode {r['decode_tok_s']:6.1f} tok/s  tokens {r['completion_tokens']:3}  sha {r['sha']}  tok/round {r['tokens_per_round']:.2f}  accepted {r['accepted']}/{r['drafted']}", flush=True)
    del draft
