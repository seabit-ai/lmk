"""Three agent-shaped tasks against a running lmk (thinking off). usage: tasks.py <url> <model>  → raw/*.json + a summary."""
import json, sys, urllib.request
from pathlib import Path

URL, MODEL = sys.argv[1], sys.argv[2]
RAW = Path(__file__).parent / "raw"; RAW.mkdir(exist_ok=True)
TOOLS = [
 {"type": "function", "function": {"name": "file_read", "description": "Read a text file and return its contents.",
  "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}}, "required": ["path"]}}},
 {"type": "function", "function": {"name": "file_edit", "description": "Replace old_text with new_text in a file. old_text must appear exactly once.",
  "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
]
SYSTEM = {"role": "system", "content": "You are kitten, a coding agent. Use tools when needed."}

def chat(name, messages):
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "temperature": 0, "max_tokens": 2000}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    (RAW / f"{name}.json").write_text(json.dumps(d, indent=1))
    m = d["choices"][0]["message"]
    calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in (m.get("tool_calls") or [])]
    return m, calls, d["usage"]["completion_tokens"]

# T1 two steps: read, then edit based on what was read
ask = {"role": "user", "content": "In notes.md, change the milk item to say 'buy almond milk'. Read the file first."}
m, calls, n = chat("t1-step1", [SYSTEM, ask])
print("T1 step1:", calls, "| tokens", n)
if calls and calls[0][0] == "file_read":
    tc = m["tool_calls"][0]
    m2, calls2, n2 = chat("t1-step2", [SYSTEM, ask, {"role": "assistant", "content": None, "tool_calls": [tc]},
                                       {"role": "tool", "tool_call_id": tc["id"], "content": "# notes\n- ship lmk\n- buy oat milk\n- call mom"}])
    print("T1 step2:", calls2, "| tokens", n2)
# T2 no tool needed
m, calls, n = chat("t2", [SYSTEM, {"role": "user", "content": "In one sentence, what does 'idempotent' mean?"}])
print("T2:", calls, "|", (m.get("content") or "")[:120].replace("\n", " "), "| tokens", n)
# T3 integer argument
m, calls, n = chat("t3", [SYSTEM, {"role": "user", "content": "Show me the first 5 lines of notes.md."}])
print("T3:", calls, "| types:", [type(v).__name__ for _, a in calls for v in a.values()], "| tokens", n)
