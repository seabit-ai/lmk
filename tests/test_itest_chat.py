"""Integration: the real engine and the real model behind the real HTTP surface.
LMK_ITEST=1; LMK_ITEST_MODEL overrides the model directory."""
import json
import os
import threading
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.itest

from itest_model import model_dir
TOOLS = [{"type": "function", "function": {
    "name": "file_read", "description": "Read a text file and return its contents.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}},
                   "required": ["path"]}}}]


@pytest.fixture(scope="module")
def server():
    from lmk.engine import MlxEngine
    from lmk.server import LmkServer

    srv = LmkServer(MlxEngine("itest-model", model_dir(), 32768), "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def chat(srv, messages, **extra):
    body = {"model": "itest-model", "stream": True, "messages": messages, **extra}
    req = urllib.request.Request(f"http://127.0.0.1:{srv.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Lmk-Purpose": "itest"}, method="POST")
    raw = urllib.request.urlopen(req, timeout=600).read().decode()
    chunks = [json.loads(l[6:]) for l in raw.split("\n") if l.startswith("data: ") and l != "data: [DONE]"]
    pick = lambda key: "".join(c["choices"][0]["delta"].get(key) or "" for c in chunks if c["choices"])
    calls = [c["choices"][0]["delta"]["tool_calls"][0] for c in chunks if c["choices"] and c["choices"][0]["delta"].get("tool_calls")]
    usage = next(c["usage"] for c in chunks if c.get("usage"))
    prefill = [c["lmk"]["prefill"] for c in chunks if c.get("object") == "lmk.prefill"]
    return {"content": pick("content"), "reasoning": pick("reasoning_content"), "calls": calls, "usage": usage, "prefill": prefill}


# Same acceptance as kitten's lmstudio provider itest: call out, result back, text answer.
def test_tool_call_round_trip(server):
    system = {"role": "system", "content": "You are kitten, a coding agent. Use tools when needed."}
    ask = {"role": "user", "content": "Read the first 20 lines of notes.md and tell me what the second bullet says."}
    first = chat(server, [system, ask], tools=TOOLS)
    assert len(first["calls"]) == 1
    call = first["calls"][0]
    assert call["function"]["name"] == "file_read"
    args = json.loads(call["function"]["arguments"])
    assert args["path"] == "notes.md" and args.get("max_lines", 20) == 20  # an integer, not "20"
    assert first["reasoning"] and "think>" not in first["content"]

    second = chat(server, [system, ask,
                           {"role": "assistant", "content": None, "tool_calls": [{k: v for k, v in call.items() if k != "index"}]},
                           {"role": "tool", "tool_call_id": call["id"], "content": "# notes\n- ship lmk\n- buy oat milk"}],
                  tools=TOOLS)
    assert "oat milk" in second["content"].lower()
    assert second["calls"] == []


# The number LM Studio never gave us (wish list WISH-002): cache hits, in usage.
def test_second_turn_reports_its_cache_hits_in_usage(server):
    system = {"role": "system", "content": "".join(f"Project rule {i}: answer in one short sentence. " for i in range(250))}
    turn1 = [system, {"role": "user", "content": "Say hello."}]
    first = chat(server, turn1)
    second = chat(server, turn1 + [{"role": "assistant", "content": first["content"]},
                                   {"role": "user", "content": "Now say goodbye."}])
    hits = second["usage"]["prompt_tokens_details"]["cached_tokens"]
    # restore lands on the largest checkpointed 256-token boundary inside the shared prefix (research LMK-002)
    assert hits >= first["usage"]["prompt_tokens"] - (2048 + 256)
    assert second["prefill"][0]["cached"] == hits, "the first progress chunk announces the same number"


# WISH-019 acceptance: after a warm-up, the first real request restores the prefix.
def test_warmup_makes_the_first_real_request_hit(server):
    system = {"role": "system", "content": "".join(f"Warm rule {i}: keep answers to one short sentence. " for i in range(250))}
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}/lmk/v1/warmup",
                                 data=json.dumps({"model": "itest-model", "messages": [system]}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    warm = json.loads(urllib.request.urlopen(req, timeout=600).read())
    assert warm["cached_tokens"] == 0, "a fresh prefix starts cold"

    real = chat(server, [system, {"role": "user", "content": "Say hello."}])
    hits = real["usage"]["prompt_tokens_details"]["cached_tokens"]
    print(f"warm prompt={warm['prompt_tokens']} real prompt={real['usage']['prompt_tokens']} hits={hits}")
    assert hits >= warm["prompt_tokens"] - (2048 + 256)


# Image input end to end: a generated PNG with a number only the pixels carry.
def test_the_model_reads_an_image(server):
    import base64
    import io

    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 240), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 120)
    except OSError:
        font = ImageFont.load_default()
    draw.text((60, 50), "4217", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    out = chat(server, [{"role": "user", "content": [
        {"type": "text", "text": "What number is written in this image? Answer with the number only."},
        {"type": "image_url", "image_url": {"url": url}}]}])
    print("image answer:", repr(out["content"]), "prompt_tokens:", out["usage"]["prompt_tokens"])
    assert "4217" in out["content"]
