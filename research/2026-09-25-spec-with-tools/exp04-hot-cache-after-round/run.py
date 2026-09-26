"""exp04: does a row that finishes inside a round hand off a hot cache shorter than its all_tokens,
and does the next turn of a tools conversation hit it?
usage (lmk checkout): ENGINE=<engine dir> PYTHONPATH=$ENGINE:. .venv/bin/python <this> <out_dir>"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp01-tools-draft-sha"))
from run import MESSAGES, TOOLS  # the fizzbuzz agent prompt

from mlx_engine.generate import create_generator
from lmk.chat import CallerIdentity, run_chat
from lmk.config import load_config
from lmk.engine import MlxEngine
from lmk.models import resolve_draft, resolve_model

restores = []


class Grab(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith("Prompt cache restore"):
            restores.append(msg)


def kv_offsets(cache):
    return sorted({int(getattr(c, "offset")) for c in cache if isinstance(getattr(c, "offset", None), int)})


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("mlx_engine").addHandler(Grab())
    source = load_config().model.source
    engine = MlxEngine("exp04", resolve_model(source).path, 32768,
                       template_kwargs={"enable_thinking": True, "reasoning_effort": "low"}, kv_cache_bits=8,
                       draft_path=resolve_draft(source, None), draft_kind="dflash2")
    toggle = {"spec": True}
    generate = engine.generate
    engine.generate = lambda prompt, **kw: generate(prompt, **{**kw, "sampling": {**(kw.get("sampling") or {}),
                                                                                    "speculative_decoding_toggle": toggle["spec"]}})
    kit, coord = engine._kit, engine._kit._prompt_cache_coordinator
    report = {}

    before = engine._drafter_counters()
    t1 = run_chat(engine, {"messages": MESSAGES, "tools": TOOLS, "temperature": 0, "max_tokens": 400},
                  CallerIdentity(purpose="exp04"), emit=lambda c: None)
    after = engine._drafter_counters()
    entry = coord._hot_entry
    report["turn1"] = dict(finish=t1["finish_reason"], calls=[c["function"]["name"] for c in t1["tool_calls"]],
                           drafted=after[2] - before[2], accepted=after[1] - before[1],
                           hot_tokens=None if entry is None else len(entry.prompt_input_ids),
                           hot_kv_offsets=None if entry is None else kv_offsets(entry.prompt_cache),
                           hot_last_tokens=None if entry is None else kit.tokenizer.decode(entry.prompt_input_ids[-3:]))
    print("turn1", json.dumps(report["turn1"]), flush=True)

    call = t1["tool_calls"][0]
    history = MESSAGES + [{"role": "assistant", "content": t1["content"] or None, "reasoning_content": t1["reasoning_content"],
                           "tool_calls": [{k: v for k, v in call.items() if k != "index"}]},
                          {"role": "tool", "tool_call_id": call["id"], "content": "wrote fizzbuzz.py (412 bytes)"}]
    prompt2 = engine.chat_format().render(history, TOOLS)
    tokens2 = kit.tokenizer.encode(prompt2, add_special_tokens=False)
    hot = entry.prompt_input_ids if entry is not None else []
    common = next((i for i, (a, b) in enumerate(zip(hot, tokens2)) if a != b), min(len(hot), len(tokens2)))
    report["turn2_prompt"] = dict(tokens=len(tokens2), common_with_hot=common, hot_tokens=len(hot),
                                  full_prefix=common == len(hot),
                                  hot_tail=kit.tokenizer.decode(hot[max(0, common - 3):common + 3]),
                                  prompt_at=kit.tokenizer.decode(tokens2[max(0, common - 3):common + 3]))
    print("turn2 prompt", json.dumps(report["turn2_prompt"]), flush=True)

    def turn2(name):
        restores.clear()
        text, top = "", None
        for r in create_generator(kit, tokens2, temp=0.0, max_tokens=60, top_logprobs=5, request_id=name,
                                  speculative_decoding_toggle=False):
            if top is None and r.top_logprobs:
                top = [(t.text, round(t.logprob, 4)) for t in r.top_logprobs[0]]
            text += r.text
        return dict(text=text, first_top5=top, restore=list(restores))

    toggle["spec"] = False
    report["A_hot"] = turn2("A-hot")                    # the hot entry turn 1 left behind
    print("A hot ", json.dumps(report["A_hot"], ensure_ascii=False), flush=True)
    # B: the same prompt with no hot entry: the disk snapshots, stored at honest lengths
    run_chat(engine, {"messages": [{"role": "user", "content": "Say hi."}], "temperature": 0, "max_tokens": 4},
             CallerIdentity(purpose="exp04"), emit=lambda c: None)   # replaces the hot entry with an unrelated one
    report["B_disk"] = turn2("B-disk")
    print("B disk", json.dumps(report["B_disk"], ensure_ascii=False), flush=True)
    a = dict(report["A_hot"]["first_top5"]); b = dict(report["B_disk"]["first_top5"])
    shared = set(a) & set(b)
    report["first_logprob_max_abs_diff"] = max(abs(a[k] - b[k]) for k in shared) if shared else None
    report["same_text"] = report["A_hot"]["text"] == report["B_disk"]["text"]
    print("A vs B: same text", report["same_text"], "max |dlogprob| over shared top-5", report["first_logprob_max_abs_diff"], flush=True)
    (out / "results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    engine.close()


if __name__ == "__main__":
    main()
