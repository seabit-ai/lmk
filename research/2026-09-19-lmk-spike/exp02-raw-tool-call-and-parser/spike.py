"""exp02: what does the model literally write for a tool call, and does mlx-lm's
qwen3_coder parser turn it into typed JSON?"""
import sys, json, re
from mlx_engine.generate import load_model, create_generator, tokenize
from mlx_lm.tool_parsers import qwen3_coder

TOOLS = [
    {"type": "function", "function": {"name": "file_read", "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "max_lines": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep_search", "description": "Search files for a pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}},
            "case_sensitive": {"type": "boolean"}}, "required": ["pattern"]}}},
]
kit = load_model(sys.argv[1], max_kv_size=32768, max_seq_nums=4)
msgs = [{"role": "system", "content": "You are kitten, a coding agent. Use tools when needed."},
        {"role": "user", "content": "Read the first 20 lines of notes.md, and also search for the word teapot, "
                                    "case-insensitively, in the docs and src directories."}]
text = kit.tokenizer.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True)
open("rendered-prompt.txt", "w").write(text)
raw = "".join(r.text for r in create_generator(kit, tokenize(kit, text), max_tokens=700, temp=0.0, request_id="lmk-exp02"))
open("raw-output.txt", "w").write(raw)

answer = raw.split("</think>", 1)[-1]
blocks = re.findall(re.escape(qwen3_coder.tool_call_start) + r"(.*?)" + re.escape(qwen3_coder.tool_call_end), answer, re.DOTALL)
parsed = []
for b in blocks:
    try: parsed.append(qwen3_coder.parse_tool_call(b.strip(), TOOLS))
    except Exception as e: parsed.append({"PARSE_ERROR": repr(e), "block": b[:200]})
print(json.dumps({"think_closed": "</think>" in raw, "n_blocks": len(blocks), "parsed": parsed}, ensure_ascii=False, indent=1))
