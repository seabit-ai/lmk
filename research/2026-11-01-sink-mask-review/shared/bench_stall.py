"""exp04: how much a decoding request stalls while another request's long prompt is read, by prefill chunk.

usage: bench_stall.py --kv-bits {16,8} --step N --prompt-tokens N --out raw/<arm>.jsonl   (PYTHONPATH=<engine worktree>:<lmk>)
Loads the catalog 27B-4bit as lmk does (batched kit, max_seq_nums=2), warms up, then:
  A: a short prompt, decodes continuously; every token's arrival time is recorded.
  B: starts once A has produced 10 tokens; a long prompt (filler), prefill measured to its first token.
A is stopped when B's first token arrives. The report is A's inter-token gaps before B started (baseline)
and while B was reading its prompt (stall), plus B's prefill seconds (compare exp02's solo numbers)."""
import argparse, glob, json, os, statistics, threading, time
from pathlib import Path

import mlx.core as mx
from mlx_engine.generate import create_generator, load_model
from lmk.engine import limit_buffer_cache

MAIN = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--lmstudio-community--Qwen3.8-27B-MLX-4bit/snapshots/*")))[-1]
FILLER = "The crew logged routine progress on the retaining wall today and the inspector signed the sheet. "


def chat_tokens(kit, text):
    s = kit.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                          tokenize=False, enable_thinking=False)
    return kit.tokenizer.encode(s, add_special_tokens=False)


def long_prompt(kit, n_tokens):
    per = (len(chat_tokens(kit, FILLER * 200)) - len(chat_tokens(kit, FILLER * 100))) / 100
    return chat_tokens(kit, FILLER * int(n_tokens / per) + "\n\nIn one sentence: what did the crew work on?")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kv-bits", type=int, required=True)
    ap.add_argument("--step", type=int, required=True); ap.add_argument("--prompt-tokens", type=int, required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    limit_buffer_cache(mx)
    kit = load_model(Path(MAIN), max_kv_size=65536, max_seq_nums=2, kv_bits=None if a.kv_bits == 16 else a.kv_bits,
                     prefill_step_size=a.step, enable_disk_cache=False)
    arm = f"kv{a.kv_bits}-step{a.step}-B{a.prompt_tokens}"
    # warm-up: one short request so A's first tokens are not the cold start
    for _ in create_generator(kit, chat_tokens(kit, "Say hello."), max_tokens=8, temp=0.0, request_id="warm"): pass

    a_tokens = chat_tokens(kit, "Write a long story about a lighthouse keeper. Keep going until told to stop.")
    b_tokens = long_prompt(kit, a.prompt_tokens)
    t0 = time.perf_counter(); a_times = []; b_start = [None]; b_first = [None]; stop = threading.Event(); b_started = threading.Event()

    def run_a():
        gen = create_generator(kit, list(a_tokens), max_tokens=100000, temp=0.0, request_id="A")
        try:
            for r in gen:
                now = time.perf_counter() - t0
                a_times.extend([now] * len(r.tokens))
                if len(a_times) >= 10: b_started.set()
                if stop.is_set() or r.stop_condition: break
        finally:
            gen.close()

    def run_b():
        b_started.wait()
        b_start[0] = time.perf_counter() - t0
        gen = create_generator(kit, list(b_tokens), max_tokens=4, temp=0.0, request_id="B")
        try:
            for r in gen:
                if r.tokens and b_first[0] is None:
                    b_first[0] = time.perf_counter() - t0; stop.set()
                if r.stop_condition: break
        finally:
            gen.close()

    ta = threading.Thread(target=run_a); tb = threading.Thread(target=run_b); ta.start(); tb.start(); ta.join(); tb.join()
    gaps = [b - a_ for a_, b in zip(a_times, a_times[1:])]
    before = [g for g, t in zip(gaps, a_times[1:]) if t <= b_start[0]]
    during = [g for g, t in zip(gaps, a_times[1:]) if b_start[0] < t <= b_first[0]]
    row = dict(arm=arm, prompt_tokens_B=len(b_tokens), b_prefill_s=round(b_first[0] - b_start[0], 1),
               a_tokens_before=len(before) + 1, a_gap_before_ms=round(statistics.median(before) * 1000) if before else None,
               a_tokens_during=len(during), a_gap_during_median_ms=round(statistics.median(during) * 1000) if during else None,
               a_gap_during_max_ms=round(max(during) * 1000) if during else None,
               a_tok_s_during=round(len(during) / (b_first[0] - b_start[0]), 2))
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n"); f.write(json.dumps(dict(arm=arm, a_times=[round(t, 3) for t in a_times], b_start=round(b_start[0], 3), b_first=round(b_first[0], 3))) + "\n")


if __name__ == "__main__":
    main()
