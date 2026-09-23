"""The hand-written categories: instruction following (format constraints a program can check)
and multi-step tool tasks over an in-memory file system."""
import json, re

# --- instruct: (id, prompt, checker(content) -> bool)
def _lines(s): return [l for l in s.strip().splitlines() if l.strip()]
INSTRUCT = [
    ("three-bullets", "Name three risks of running a database without backups. Answer with exactly three bullet points, each starting with '- ', and nothing else.",
     lambda s: len(_lines(s)) == 3 and all(l.startswith("- ") for l in _lines(s))),
    ("json-object", "Describe a fictional person as a JSON object with exactly these keys: name (string), age (integer), city (string). Output only the JSON, no code fence, no prose.",
     lambda s: (lambda o: isinstance(o, dict) and set(o) == {"name", "age", "city"} and isinstance(o["age"], int) and isinstance(o["name"], str))(_json(s))),
    ("five-words", "Explain what a compiler does in exactly five words.",
     lambda s: len(re.findall(r"[A-Za-z0-9'-]+", s)) == 5),
    ("no-letter-e", "Describe the ocean in two or three sentences without using the letter 'e' anywhere in your answer.",
     lambda s: "e" not in s.lower() and len(s.split()) >= 8),
    ("result-tags", "What is the capital of Australia? Put your answer inside <result></result> tags and output nothing outside the tags.",
     lambda s: re.fullmatch(r"\s*<result>\s*Canberra\s*</result>\s*", s, re.I) is not None),
    ("all-caps", "Write one sentence about mountains in ALL CAPITAL LETTERS.",
     lambda s: s == s.upper() and any(c.isalpha() for c in s) and len(s.split()) >= 4),
    ("countdown", "Output the integers from 10 down to 1 as a single comma-separated line with no spaces, and nothing else.",
     lambda s: s.strip() == "10,9,8,7,6,5,4,3,2,1"),
    ("ends-with-done", "Give two tips for sleeping better. Your answer must end with the single word 'Done.' as the last word.",
     lambda s: s.strip().endswith("Done.") and len(s.split()) >= 10),
    ("sorted-fruits", "List four fruits in alphabetical order, one per line, all lowercase, no numbering or bullets, nothing else.",
     lambda s: len(_lines(s)) == 4 and _lines(s) == sorted(_lines(s)) and all(l == l.lower() and re.fullmatch(r"[a-z ]+", l) for l in _lines(s))),
    ("yes-or-no", "Is 91 a prime number? Answer with exactly one word: yes or no.",
     lambda s: s.strip().strip(".").lower() == "no"),
]
def _json(s):
    try: return json.loads(s.strip().strip("`").removeprefix("json").strip())
    except Exception: return None

# --- tools: each task = (id, files, user prompt, checker(final_files, final_answer) -> bool)
TOOLS_SPEC = [
 {"type": "function", "function": {"name": "file_list", "description": "List the files in the working directory.", "parameters": {"type": "object", "properties": {}}}},
 {"type": "function", "function": {"name": "file_read", "description": "Read a text file and return its contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
 {"type": "function", "function": {"name": "file_write", "description": "Write the whole file (create or overwrite).", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
 {"type": "function", "function": {"name": "file_edit", "description": "Replace old_text with new_text in a file. old_text must appear exactly once.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
]
def run_tool(files, name, args):
    """Returns the tool's text result; mutates files."""
    try:
        if name == "file_list": return "\n".join(sorted(files))
        if name == "file_read":
            return files[args["path"]] if args["path"] in files else f"error: no such file: {args['path']}"
        if name == "file_write": files[args["path"]] = args["content"]; return "ok"
        if name == "file_edit":
            p = args["path"]
            if p not in files: return f"error: no such file: {p}"
            if files[p].count(args["old_text"]) != 1: return "error: old_text must appear exactly once"
            files[p] = files[p].replace(args["old_text"], args["new_text"]); return "ok"
    except (KeyError, TypeError) as e:
        return f"error: bad arguments: {e}"
    return f"error: unknown tool {name}"

TOOL_TASKS = [
    ("edit-after-read", {"notes.md": "# notes\n- ship lmk\n- buy oat milk\n- call mom\n"},
     "In notes.md, change the milk item to say 'buy almond milk'. Read the file first.",
     lambda f, a: f.get("notes.md") == "# notes\n- ship lmk\n- buy almond milk\n- call mom\n"),
    ("find-todo", {"a.py": "def f():\n    return 1\n", "b.py": "def g():\n    # TODO: handle None\n    return 2\n", "c.py": "x = 3\n"},
     "One of the Python files in the working directory contains a TODO comment. Which file is it? Answer with just the file name.",
     lambda f, a: "b.py" in a and "a.py" not in a and "c.py" not in a),
    ("report-port", {"config.json": json.dumps({"service": {"name": "api", "listen": {"host": "0.0.0.0", "port": 8443}}, "debug": False}, indent=2)},
     "What port does the service in config.json listen on? Answer with just the number.",
     lambda f, a: re.search(r"\b8443\b", a) is not None),
    ("append-line", {"todo.md": "- water plants\n- pay rent\n"},
     "Add the item '- call the dentist' to the end of todo.md, keeping the existing items.",
     lambda f, a: f.get("todo.md", "").rstrip("\n").splitlines() == ["- water plants", "- pay rent", "- call the dentist"]),
    ("count-lines", {"x.txt": "one\ntwo\nthree\n", "y.txt": "four\nfive\nsix\nseven\n"},
     "How many lines are there in total across x.txt and y.txt? Answer with just the number.",
     lambda f, a: re.search(r"\b7\b", a) is not None),
]
