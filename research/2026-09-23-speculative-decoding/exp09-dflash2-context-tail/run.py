"""exp09: DFlash 2 acceptance vs how much of the prompt's hidden states the drafter gets.
usage: run.py <out_dir>   (lmk .venv, mlx-vlm 0.6.16; no engine code)"""
import glob, hashlib, json, os, sys, time
from pathlib import Path
import mlx.core as mx
import mlx_vlm
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.speculative.dflash import _dflash_rounds
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.speculative.common import speculative_stats_snapshot, speculative_stats_since

MAIN = "lmstudio-community/Qwen3.8-27B-MLX-4bit"; DRAFT = "incoai/Qwen3.8-27B-DFlash2"
EOS = {248046, 248044}; MAX_TOKENS = 400
SENTENCE = "The quick brown fox jumps over the lazy dog. "
REQUEST = "Write a Python class implementing an LRU cache with get and put, with type hints and docstrings."

def snap(repo):
    return Path(sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--" + repo.replace("/", "--") + "/snapshots/*")), key=os.path.getmtime)[-1])

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
model = mlx_vlm.utils.load_model(snap(MAIN), lazy=False, trust_remote_code=False); lm = model.language_model
tok = getattr(mlx_vlm.utils.load_processor(snap(MAIN)), "tokenizer")
draft, kind = load_drafter(str(snap(DRAFT)))
text = tok.apply_chat_template([{"role": "user", "content": f"Background text (ignore it):\n{SENTENCE * 400}\n\n{REQUEST}"}],
                               add_generation_prompt=True, tokenize=False, enable_thinking=False)
toks = tok.encode(text, add_special_tokens=False); T = len(toks)
print(f"prompt {T} tokens", flush=True)
argmax = lambda logits: mx.argmax(logits, axis=-1).astype(mx.int32)
layer_ids = list(draft.config.target_layer_ids)

def decode_plain():
    ids = mx.array([toks], dtype=mx.int32); outp = []; t0 = time.perf_counter(); first = None
    for token, _ in generate_step(ids, model, None, None, max_tokens=MAX_TOKENS, temperature=0.0, prefill_step_size=512):
        t = int(token.item()) if hasattr(token, "item") else int(token)
        if first is None: first = time.perf_counter()
        if t in EOS: break
        outp.append(t)
    return outp, (len(outp) - 1) / (time.perf_counter() - first)

def decode_dflash(ctx_tokens):
    cache = make_prompt_cache(lm); ids = mx.array([toks], dtype=mx.int32)
    hidden_parts = []
    for s in range(0, T, 512):   # prefill in chunks, capturing the drafter's target layers
        o = lm(ids[:, s:s + 512], cache=cache, capture_layer_ids=layer_ids)
        hidden_parts.append(mx.concatenate(o.hidden_states, axis=-1)); logits = o.logits
    mx.eval(hidden_parts, logits)
    hidden = mx.concatenate(hidden_parts, axis=1)[:, -ctx_tokens:, :] if ctx_tokens else mx.concatenate(hidden_parts, axis=1)
    b = int(argmax(logits[:, -1, :]).item()); outp = [b]
    snapshot = speculative_stats_snapshot(draft); t0 = time.perf_counter()
    for token, _ in _dflash_rounds(model, draft, cache, hidden, first_bonus=b, max_tokens=MAX_TOKENS, sampler=argmax, greedy_sampling=True):
        t = int(token.item()) if hasattr(token, "item") else int(token)
        if t in EOS: break
        outp.append(t)
    dt = time.perf_counter() - t0
    rounds, accepted, drafted = speculative_stats_since(draft, snapshot)
    return outp, (len(outp) - 1) / dt, rounds, accepted, drafted

base, base_rate = decode_plain(); base_sha = hashlib.sha1(tok.decode(base).encode()).hexdigest()[:10]
print(f"plain           {base_rate:6.1f} tok/s  tokens {len(base)}  sha {base_sha}", flush=True)
json.dump(dict(arm="plain", tok_s=base_rate, tokens=len(base), sha=base_sha, text=tok.decode(base)), open(out / "plain.json", "w"), indent=1)
for ctx in (None, 1024, 256, 32):
    label = "full" if ctx is None else f"last{ctx}"
    o, rate, rounds, accepted, drafted = decode_dflash(ctx)
    sha = hashlib.sha1(tok.decode(o).encode()).hexdigest()[:10]
    same = "identical" if sha == base_sha else "DIFFERENT"
    print(f"dflash2 {label:8} {rate:6.1f} tok/s  x{rate/base_rate:.2f}  tokens {len(o)}  sha {sha} {same}  tok/round {len(o)/max(1,rounds):.2f}  accepted {accepted}/{drafted}", flush=True)
    json.dump(dict(arm=label, ctx_tokens=ctx or T, tok_s=rate, tokens=len(o), sha=sha, rounds=rounds, accepted=accepted, drafted=drafted, text=tok.decode(o)),
              open(out / f"dflash2-{label}.json", "w"), indent=1)
