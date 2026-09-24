"""usage: run.py <arm> <url> <model> [category ...]   → raw/<arm>/<category>/<id>-run<k>.json
       run.py grade                                    → results table from raw/"""
import json, os, re, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tasks import INSTRUCT, TOOL_TASKS, TOOLS_SPEC, run_tool

ROOT = Path(__file__).resolve().parent.parent; RAW = ROOT / "raw"; DATA = ROOT / "data"
RUNS, MAX_TOKENS, PARALLEL, MAX_TOOL_STEPS = int(os.environ.get("EVAL_RUNS", 3)), 8000, 2, 8
LIMIT = int(os.environ.get("EVAL_LIMIT", 0))   # smoke test: at most this many items per category

def chat(url, model, messages, tools=None):
    body = {"model": model, "messages": messages, "max_tokens": MAX_TOKENS}
    if tools: body["tools"] = tools
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Lmk-Purpose": "eval"})
    t0 = time.monotonic(); d = json.loads(urllib.request.urlopen(req, timeout=3600).read()); ms = int((time.monotonic() - t0) * 1000)
    m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "tool_calls": m.get("tool_calls") or [],
            "finish": d["choices"][0]["finish_reason"], "tokens": d["usage"]["completion_tokens"], "ms": ms, "raw_message": m}

# --- the four categories: each yields (id, run_fn(url, model) -> record with 'pass')
def math_items():
    for q in map(json.loads, open(DATA / "gsm8k_50.jsonl")):
        def run(url, model, q=q):
            r = chat(url, model, [{"role": "user", "content": q["question"] + "\n\nSolve it step by step, then give the final answer on the last line in the form `Answer: <number>` with nothing after the number."}])
            got = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", r["content"]); got = got[-1].replace(",", "") if got else None
            want = q["answer"].replace(",", "")
            r["pass"] = got is not None and r["finish"] != "length" and abs(float(got) - float(want)) < 1e-6
            r["got"], r["want"] = got, want; return r
        yield q["id"], run

def code_items():
    for t in map(json.loads, open(DATA / "humaneval_40.jsonl")):
        def run(url, model, t=t):
            r = chat(url, model, [{"role": "user", "content": "Complete the following Python function. Reply with the complete function (signature and body, plus any imports it needs) in a single ```python code block and nothing else.\n\n```python\n" + t["prompt"] + "```"}])
            blocks = re.findall(r"```python\n(.*?)```", r["content"], re.S)
            code = blocks[-1] if blocks else r["content"]
            r["pass"] = r["finish"] != "length" and _run_tests(code, t["test"], t["entry_point"]); return r
        yield t["id"].replace("/", "-"), run

def _run_tests(code, test, entry_point):
    with tempfile.TemporaryDirectory() as d:
        Path(d, "sol.py").write_text(code + "\n\n" + test + f"\n\ncheck({entry_point})\n")
        try:
            return subprocess.run([sys.executable, "-I", "sol.py"], cwd=d, capture_output=True, timeout=10).returncode == 0
        except subprocess.TimeoutExpired:
            return False

def instruct_items():
    for iid, prompt, check in INSTRUCT:
        def run(url, model, prompt=prompt, check=check):
            r = chat(url, model, [{"role": "user", "content": prompt}])
            try: r["pass"] = r["finish"] != "length" and bool(check(r["content"]))
            except Exception: r["pass"] = False
            return r
        yield iid, run

def tools_items():
    for tid, files0, prompt, check in TOOL_TASKS:
        def run(url, model, files0=files0, prompt=prompt, check=check):
            files = dict(files0); msgs = [{"role": "system", "content": "You are a coding agent working in a small directory. Use the tools to look at files before answering or editing."},
                                          {"role": "user", "content": prompt}]
            steps, tokens, ms, finish = [], 0, 0, "stop"
            for _ in range(MAX_TOOL_STEPS):
                r = chat(url, model, msgs, tools=TOOLS_SPEC); tokens += r["tokens"]; ms += r["ms"]; finish = r["finish"]
                steps.append({k: r[k] for k in ("content", "reasoning", "tool_calls", "finish", "tokens", "ms")})
                if not r["tool_calls"] or finish == "length": break
                msgs.append(r["raw_message"])
                for c in r["tool_calls"]:
                    try: args = json.loads(c["function"]["arguments"] or "{}")
                    except json.JSONDecodeError: args = {}
                    msgs.append({"role": "tool", "tool_call_id": c["id"], "content": run_tool(files, c["function"]["name"], args)})
            answer = steps[-1]["content"]
            return {"pass": finish != "length" and bool(check(files, answer)), "steps": steps, "tokens": tokens, "ms": ms, "finish": finish, "final_files": files, "content": answer}
        yield tid, run

CATEGORIES = {"math": math_items, "code": code_items, "instruct": instruct_items, "tools": tools_items}

def run_arm(arm, url, model, cats):
    jobs = []
    for cat in cats:
        for n, (iid, fn) in enumerate(CATEGORIES[cat]()):
            if LIMIT and n >= LIMIT: break
            for k in range(1, RUNS + 1):
                out = RAW / arm / cat / f"{iid}-run{k}.json"
                if not out.exists(): jobs.append((cat, iid, k, fn, out))
    print(f"{arm}: {len(jobs)} requests to run", flush=True)
    def do(job):
        cat, iid, k, fn, out = job
        try: rec = fn(url, model)
        except Exception as e: rec = {"pass": False, "error": repr(e), "tokens": 0, "ms": 0, "finish": "error"}
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        print(f"{arm} {cat} {iid} run{k}: {'PASS' if rec['pass'] else 'fail'} {rec.get('finish')} {rec.get('tokens')}tok {rec.get('ms', 0)//1000}s", flush=True)
    with ThreadPoolExecutor(PARALLEL) as ex: list(ex.map(do, jobs))

def grade():
    arms = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    print("| arm | " + " | ".join(f"{c} pass" for c in CATEGORIES) + " | ran out of tokens | tokens/answer (median) | s/answer (median) |")
    print("|---|" + "---|" * (len(CATEGORIES) + 3))
    for arm in arms:
        cells, ran_out, toks, secs = [], 0, [], []
        for cat in CATEGORIES:
            recs = [json.loads(p.read_text()) for p in sorted((RAW / arm / cat).glob("*.json"))] if (RAW / arm / cat).exists() else []
            if not recs: cells.append("—"); continue
            n = sum(r["pass"] for r in recs); cells.append(f"{n}/{len(recs)} ({100*n/len(recs):.0f}%)")
            ran_out += sum(r.get("finish") == "length" for r in recs); toks += [r.get("tokens", 0) for r in recs]; secs += [r.get("ms", 0)/1000 for r in recs]
        med = lambda xs: sorted(xs)[len(xs)//2] if xs else 0
        print(f"| {arm} | " + " | ".join(cells) + f" | {ran_out} | {med(toks):,.0f} | {med(secs):.0f} |")

if __name__ == "__main__":
    if sys.argv[1] == "grade": grade()
    else: run_arm(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:] or list(CATEGORIES))
