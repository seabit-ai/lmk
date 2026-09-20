"""exp01: drive the open-source mlx-engine directly — no LM Studio server involved.
Borrowed interpreter + site-packages: LM Studio's vendored Python (nothing installed)."""
import sys, time, json
from mlx_engine.generate import load_model, create_generator, tokenize
from mlx_engine.utils.prompt_progress_reporter import PromptProgressReporter

MODEL = sys.argv[1]

class Recorder(PromptProgressReporter):
    def __init__(self): self.events = []; self.t0 = time.monotonic()
    def _t(self): return round(time.monotonic() - self.t0, 2)
    def begin(self, is_draft, cached_tokens, total_prompt_tokens, prefill_tokens_processed):
        self.events.append({"ev": "begin", "t": self._t(), "cached": cached_tokens, "total": total_prompt_tokens}); return True
    def update(self, is_draft, prefill_tokens_processed):
        self.events.append({"ev": "update", "t": self._t(), "processed": prefill_tokens_processed}); return True
    def finish(self, is_draft, prefill_tokens_processed=None):
        self.events.append({"ev": "finish", "t": self._t(), "processed": prefill_tokens_processed}); return True

def chat_prompt(kit, system, user):
    tok = kit.tokenizer if hasattr(kit, "tokenizer") else None
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenize(kit, text)

t = time.monotonic()
kit = load_model(MODEL, max_kv_size=32768, max_seq_nums=4)
print(json.dumps({"step": "load", "kit": type(kit).__name__, "secs": round(time.monotonic() - t, 1)}), flush=True)

filler = "".join(f"Rule {i}: the spike document repeats itself so the prefix is long and byte-stable. " for i in range(130))
for name, question in [("A-cold", "Say the single word: alpha."), ("B-fork", "Say the single word: bravo."), ("A-again", "Say the single word: alpha.")]:
    tokens = chat_prompt(kit, filler, question)
    rec = Recorder(); out = []; t = time.monotonic(); first = None
    for r in create_generator(kit, tokens, prompt_progress_reporter=rec, max_tokens=24, temp=0.0):
        if first is None: first = round(time.monotonic() - t, 2)
        out.append(r.text)
    begin = next((e for e in rec.events if e["ev"] == "begin"), {})
    print(json.dumps({"step": name, "prompt_tokens": len(tokens), "cached": begin.get("cached"),
                      "updates": [e["processed"] for e in rec.events if e["ev"] == "update"],
                      "ttft_s": first, "total_s": round(time.monotonic() - t, 2),
                      "text": "".join(out)[:80]}), flush=True)
