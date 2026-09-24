"""Fixed-seed subsets of GSM8K test and HumanEval → data/. Run once; the files are committed."""
import gzip, io, json, random, urllib.request
from pathlib import Path
DATA = Path(__file__).resolve().parent.parent / "data"; DATA.mkdir(exist_ok=True)
rng = random.Random(20260923)
gsm = [json.loads(l) for l in urllib.request.urlopen(
    "https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/test.jsonl", timeout=60).read().decode().splitlines()]
pick = rng.sample(range(len(gsm)), 50)
with open(DATA / "gsm8k_50.jsonl", "w") as f:
    for i in sorted(pick):
        q = gsm[i]; f.write(json.dumps({"id": f"gsm8k-{i}", "question": q["question"], "answer": q["answer"].split("####")[-1].strip()}) + "\n")
raw = urllib.request.urlopen("https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz", timeout=60).read()
he = [json.loads(l) for l in gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode().splitlines()]
pick = rng.sample(range(len(he)), 40)
with open(DATA / "humaneval_40.jsonl", "w") as f:
    for i in sorted(pick):
        t = he[i]; f.write(json.dumps({"id": t["task_id"], "prompt": t["prompt"], "test": t["test"], "entry_point": t["entry_point"]}) + "\n")
print("gsm8k", 50, "humaneval", 40)
