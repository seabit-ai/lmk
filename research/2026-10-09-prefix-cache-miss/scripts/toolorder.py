import glob, os
from transformers import AutoTokenizer
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--lmstudio-community--Qwen3.8-27B-MLX-4bit/snapshots/*"))[0]
tok = AutoTokenizer.from_pretrained(snap)
text = tok.apply_chat_template(
    [{"role": "system", "content": "SYSMARKER"}, {"role": "user", "content": "hi"}],
    tools=[{"type": "function", "function": {"name": "toolmarker", "description": "d", "parameters": {}}}],
    tokenize=False, add_generation_prompt=True, reasoning_effort="low")
print("tools at char", text.find("toolmarker"), "| system at char", text.find("SYSMARKER"))
print(text[:300].replace("\n", "\\n"))
