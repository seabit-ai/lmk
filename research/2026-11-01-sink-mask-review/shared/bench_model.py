"""exp05 (a copy of exp02/03 bench.py with --model/--name): peak Metal memory and prefill time on lmk's batched kv8 path, by prefill chunk size.

usage: bench.py --kv-bits {16,8} --step N --out raw/<arm>.jsonl   (PYTHONPATH=<engine worktree>:<lmk>)
Loads the catalog 27B-4bit the way lmk does (BatchedVisionModelKit, max_seq_nums=2, auto context fit,
lmk's buffer-cache limit), then for each prompt length: reset the peak, one greedy request of 64 tokens,
record prefill seconds (time to first token), the process's peak Metal bytes, decode tok/s.
One process per arm so the peak is that arm's alone."""
import argparse, glob, json, os, sys, time
from pathlib import Path

import mlx.core as mx
from mlx_engine.generate import create_generator, load_model
from lmk.engine import limit_buffer_cache

MAIN = None  # set by --model
LENGTHS = (4_000, 16_000, 30_000, 50_000)
FILLER = "The crew logged routine progress on the retaining wall today and the inspector signed the sheet. "
QUESTION = "\n\nIn one sentence: what did the crew work on?"
MAX_TOKENS = 64


def prompt_tokens(kit, n_tokens):
    # size the filler by measured tokens-per-repeat, then land within ~1% of the target
    def encode(reps):
        text = FILLER * reps + QUESTION
        s = kit.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
        return kit.tokenizer.encode(s, add_special_tokens=False)
    per = (len(encode(200)) - len(encode(100))) / 100
    toks = encode(int(n_tokens / per))
    return toks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kv-bits", type=int, required=True)
    ap.add_argument("--step", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
    a = ap.parse_args()
    limit_buffer_cache(mx)
    t0 = time.perf_counter()
    kit = load_model(Path(a.model), max_kv_size=65536, max_seq_nums=2, kv_bits=None if a.kv_bits == 16 else a.kv_bits,
                     prefill_step_size=a.step, enable_disk_cache=False)
    load_s = time.perf_counter() - t0
    fit = getattr(kit, "_context_fit_result", None)
    info = dict(arm=f"{a.name}-kv{a.kv_bits}-step{a.step}", kit=type(kit).__name__, load_s=round(load_s, 1),
                kit_step=getattr(kit, "prefill_step_size", None), fit=None if fit is None else str(fit),
                weights_gb=round(mx.get_active_memory() / 2**30, 2), mlx=mx.__version__)
    print(json.dumps(info), flush=True)
    with open(a.out, "a") as out:
        out.write(json.dumps(info) + "\n")
        for n in LENGTHS:
            toks = prompt_tokens(kit, n)
            mx.reset_peak_memory()
            t0 = time.perf_counter(); first = None; count = 0; text = ""
            for r in create_generator(kit, list(toks), max_tokens=MAX_TOKENS, temp=0.0, request_id=f"exp02-{a.kv_bits}-{a.step}-{n}"):
                if r.tokens and first is None: first = time.perf_counter()
                count += len(r.tokens); text += r.text
                if r.stop_condition: break
            end = time.perf_counter()
            row = dict(arm=info["arm"], prompt_tokens=len(toks), prefill_s=round(first - t0, 2),
                       prefill_tok_s=round(len(toks) / (first - t0)), peak_gb=round(mx.get_peak_memory() / 2**30, 2),
                       decode_tok_s=round((count - 1) / max(1e-6, end - first), 1), tokens=count, answer=text[:80])
            print(json.dumps(row), flush=True); out.write(json.dumps(row) + "\n"); out.flush()


if __name__ == "__main__":
    main()
