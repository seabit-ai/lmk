"""exp01: decode gain from a draft model on mlx-engine's sequential path (ModelKit).

usage: run.py <out_dir> [arms...]   arms default: main 4b-n2 4b-n4 2b-n2 2b-n4
"""
import json, sys, time, glob, os
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/src/lmk/.engine/mlx-engine"))
from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.generate import create_generator

def snapshot(repo):
    hub = os.path.expanduser("~/.cache/huggingface/hub/models--" + repo.replace("/", "--") + "/snapshots/*")
    paths = sorted(glob.glob(hub), key=os.path.getmtime)
    assert paths, repo
    return Path(paths[-1])

MAIN = "lmstudio-community/Qwen3.8-27B-MLX-4bit"
DRAFTS = {"4b": "mlx-community/Qwen3.5-4B-MLX-4bit", "2b": "mlx-community/Qwen3.5-2B-MLX-4bit"}
MAX_TOKENS = 400
REPS = 2

PARA = ("The lighthouse keeper had lived on the rock for ninety-one years, or so he told the fishermen who "
        "brought his suplies every second Tuesday. Nobody beleived him, of course, but nobody argued either; "
        "the sea has a way of making arguements feel small. Each evening he climbed the hundred and twelve steps, "
        "trimmed the wick that no longer needed trimming since the lamp went electric in 1974, and wrote one line "
        "in a ledger that had belonged to his father. The lines were never about the weather. They were about "
        "the ships he had not seen: the ones that passed in fog, the ones that turned back, the one that did not.")
PROMPTS = {
    "story": "Write a 250-word story about a lighthouse keeper.",
    "code": "Write a Python function `parse_duration(s: str) -> int` that turns strings like '1h30m', '45s', "
            "'2h' into a number of seconds. Include a docstring and a small pytest test function. Code only, no explanation.",
    "copyedit": "Fix the spelling mistakes in the text below and return the whole text otherwise unchanged. "
                "Output only the corrected text.\n\n" + PARA,
}

def prompt_tokens(tok, text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                tokenize=False, enable_thinking=False)
    return tok.encode(s, add_special_tokens=False)

def run_one(kit, tokens, num_draft, spec):
    kwargs = dict(max_tokens=MAX_TOKENS, temp=0.0, speculative_decoding_toggle=spec)
    if spec:
        kwargs["num_draft_tokens"] = num_draft
    t0 = time.perf_counter(); first = None; n = 0; from_draft = 0; text = ""
    for r in create_generator(kit, tokens, **kwargs):
        if r.tokens and first is None:
            first = time.perf_counter()
        n += len(r.tokens); from_draft += sum(1 for t in r.tokens if getattr(t, "from_draft", False))
        text += r.text
        if r.stop_condition:
            break
    end = time.perf_counter()
    return dict(prompt_tokens=len(tokens), completion_tokens=n, from_draft=from_draft,
                first_token_ms=(first - t0) * 1000, total_ms=(end - t0) * 1000,
                decode_tok_s=(n - 1) / max(1e-6, end - first), text=text)

def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    arms = sys.argv[2:] or ["main", "4b-n2", "4b-n4", "2b-n2", "2b-n4"]
    t = time.perf_counter()
    kit = ModelKit(snapshot(MAIN), prefill_step_size=512); kit.start()
    print(f"main loaded in {time.perf_counter()-t:.1f}s", flush=True)
    tok = kit.tokenizer
    prompts = {k: prompt_tokens(tok, v) for k, v in PROMPTS.items()}
    # warm-up
    run_one(kit, prompts["story"][:], 2, False)
    loaded_draft = None
    for arm in arms:
        if arm == "main":
            draft, n = None, 0
        else:
            draft, n = arm.split("-n"); n = int(n)
        if draft != loaded_draft:
            if loaded_draft: kit.unload_draft_model()
            if draft:
                t = time.perf_counter(); kit.load_draft_model(snapshot(DRAFTS[draft]))
                print(f"draft {draft} loaded in {time.perf_counter()-t:.1f}s", flush=True)
            loaded_draft = draft
        for name, toks in prompts.items():
            for rep in range(1, REPS + 1):
                r = run_one(kit, toks[:], n, draft is not None)
                r.update(arm=arm, prompt=name, rep=rep)
                (out / f"{arm}-{name}-run{rep}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
                acc = r["from_draft"] / max(1, r["completion_tokens"])
                print(f"{arm:6} {name:9} run{rep}  decode {r['decode_tok_s']:6.1f} tok/s  "
                      f"tokens {r['completion_tokens']:3}  from_draft {acc:4.0%}  ttft {r['first_token_ms']:6.0f} ms", flush=True)
    kit.shutdown()

if __name__ == "__main__":
    main()
