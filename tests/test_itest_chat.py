"""Integration: the real engine and the real model behind the real HTTP surface.
LMK_ITEST=1; LMK_ITEST_MODEL overrides the model directory."""
import json
import os
import threading
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.itest

from itest_model import draft_path, kv_cache_bits, model_dir
TOOLS = [{"type": "function", "function": {
    "name": "file_read", "description": "Read a text file and return its contents.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}},
                   "required": ["path"]}}}]


@pytest.fixture(scope="module")
def server():
    from lmk.engine import MlxEngine
    from lmk.server import LmkServer

    # LMK_ITEST_THINKING=off|on: the same acceptance with thinking forced either way (a server-level
    # constant, `model.thinking`); unset = the family's default (Qwen on, Gemma off)
    forced = os.environ.get("LMK_ITEST_THINKING")
    kwargs = {"enable_thinking": forced == "on"} if forced in ("on", "off") else {}
    engine = MlxEngine("itest-model", model_dir(), 32768, template_kwargs=kwargs, kv_cache_bits=kv_cache_bits(),
                       draft_path=draft_path())
    print("itest: dialect", engine.chat_format().dialect.name, "thinking", engine.thinking_enabled(),
          "kv cache bits", kv_cache_bits(), "draft", draft_path())
    srv = LmkServer(engine, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def chat(srv, messages, **extra):
    # temperature 0: these tests check what the model can do, not how it samples; since lmk took
    # the model's own sampling defaults (temp 1.0 for Qwen) a greedy run is the repeatable one
    body = {"model": "itest-model", "stream": True, "messages": messages, "temperature": 0, **extra}
    req = urllib.request.Request(f"http://127.0.0.1:{srv.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Lmk-Purpose": "itest"}, method="POST")
    raw = urllib.request.urlopen(req, timeout=600).read().decode()
    chunks = [json.loads(l[6:]) for l in raw.split("\n") if l.startswith("data: ") and l != "data: [DONE]"]
    pick = lambda key: "".join(c["choices"][0]["delta"].get(key) or "" for c in chunks if c["choices"])
    calls = [c["choices"][0]["delta"]["tool_calls"][0] for c in chunks if c["choices"] and c["choices"][0]["delta"].get("tool_calls")]
    usage = next(c["usage"] for c in chunks if c.get("usage"))
    prefill = [c["lmk"]["prefill"] for c in chunks if c.get("object") == "lmk.prefill"]
    decode = [c["lmk"]["decode"] for c in chunks if c.get("object") == "lmk.decode"]
    return {"content": pick("content"), "reasoning": pick("reasoning_content"), "calls": calls, "usage": usage,
            "prefill": prefill, "decode": decode}


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
    if server.engine.chat_format().dialect.prompt_decides_thinking:      # Qwen: the prompt settles it either way
        assert bool(first["reasoning"]) == server.engine.thinking_enabled()
    # Gemma: the model decides per turn whatever the setting says (exp05 F1, F3) — nothing to assert
    assert "think>" not in first["content"] and "<|channel>" not in first["content"]

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


# kitten design 2026-09-24-llm-progress §9: decode progress rides the stream on a real model too.
def test_decode_progress_rides_the_stream(server):
    out = chat(server, [{"role": "user", "content": "Count from 1 to 30, one number per line."}], max_tokens=200)
    print(f"decode chunks={len(out['decode'])} first={out['decode'][:1]} last={out['decode'][-1:]}")
    assert out["decode"], "at least the first-token report"
    assert out["decode"][0]["part"] in ("thinking", "answering")
    assert out["decode"][-1]["completion_tokens"] > out["decode"][0]["completion_tokens"] or len(out["decode"]) == 1


def warmup(srv, messages, ref_id):
    req = urllib.request.Request(f"http://127.0.0.1:{srv.port}/lmk/v1/warmup",
                                 data=json.dumps({"model": "itest-model", "messages": messages}).encode(),
                                 headers={"Content-Type": "application/json", "X-Lmk-Purpose": "warmup",
                                          "X-Lmk-Ref-Id": ref_id}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


# kitten design 2026-09-25-prewarm: a warmup gives the engine up to a request at the next prefill
# step, and the next warmup of the same prefix carries on from what the first one read (PW-001).
def test_a_warmup_yields_to_a_request_and_the_next_warmup_carries_on(server):
    import threading
    import time
    import uuid

    system = {"role": "system", "content": f"Nonce {uuid.uuid4().hex}. " +
              "".join(f"Yield rule {i}: answer in one short sentence. " for i in range(1200))}
    box = {}
    first = threading.Thread(target=lambda: box.update(first=warmup(server, [system], "itest/warm-1")), daemon=True)
    first.start()
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        mine = [r for r in server.status()["in_flight"] if r["ref_id"] == "itest/warm-1"]
        if mine and (mine[0].get("prefill") or {}).get("processed", 0) >= 2048:
            break
        time.sleep(0.2)
    else:
        raise AssertionError("the warmup never read its first step")

    sent = time.monotonic()
    real = chat(server, [{"role": "user", "content": "Say hello."}], max_tokens=8)
    real_s = time.monotonic() - sent
    first.join(600)
    second = warmup(server, [system], "itest/warm-2")
    print(f"first={box['first']} real_s={real_s:.1f} real_usage={real['usage']} second={second}")

    assert box["first"]["outcome"] == "yielded"
    assert real_s < 60, "the request waited at most about one prefill step, not the whole warmup"
    assert second["outcome"] == "done"
    assert second["cached_tokens"] >= 2048, "the second warmup carried on from the first one's steps"


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
