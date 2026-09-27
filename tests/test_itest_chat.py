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
    usage_chunk = next(c for c in chunks if c.get("usage"))
    prefill = [c["lmk"]["prefill"] for c in chunks if c.get("object") == "lmk.prefill"]
    decode = [c["lmk"]["decode"] for c in chunks if c.get("object") == "lmk.decode"]
    return {"content": pick("content"), "reasoning": pick("reasoning_content"), "calls": calls,
            "usage": usage_chunk["usage"], "lmk": usage_chunk.get("lmk") or {}, "prefill": prefill, "decode": decode}


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


# research 2026-09-25-spec-with-tools: a request with tools carries the engine's tool guard, and until
# the fork walked processors through speculative rounds it never drafted (SPD-022/023).
def test_a_request_with_tools_drafts_when_a_draft_is_loaded(server):
    if server.engine.draft_stats() is None:
        pytest.skip("no draft model loaded (LMK_ITEST_DRAFT)")
    write = {"type": "function", "function": {
        "name": "file_write", "description": "Write a text file, replacing it if it exists.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}}
    out = chat(server, [{"role": "system", "content": "You are kitten, a coding agent. Use tools to act."},
                        {"role": "user", "content": "Create fizzbuzz.py printing FizzBuzz for 1..30."}],
               tools=TOOLS + [write], max_tokens=400)
    print("itest: tools request drafted", out["lmk"].get("draft_drafted"), "accepted", out["lmk"].get("draft_accepted"),
          "completion", out["usage"]["completion_tokens"], "calls", [c["function"]["name"] for c in out["calls"]])
    assert [c["function"]["name"] for c in out["calls"]] == ["file_write"]
    assert out["lmk"]["draft_drafted"] > 0


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


# design 2026-09-26-sampling-seed: an answer is replayable. Same request + same seed + the same cache
# restore point, run alone, gives the same tokens; another seed gives another answer. The first request
# names no seed (lmk draws one and returns it) and sets up the cache the replays restore from. Parametrized
# with tools too: an agent request carries the engine's tool guard, whose rounds take a different walk.
@pytest.mark.parametrize("with_tools", [False, True], ids=["plain", "tools"])
def test_the_same_seed_replays_the_same_answer_and_another_seed_does_not(server, with_tools):
    import uuid

    messages = [{"role": "system", "content": f"Session {uuid.uuid4().hex}. You are a creative assistant."},
                {"role": "user", "content": "Invent a name for a new colour and describe it in two sentences."}]
    # the incident's settings (Qwen3.8's generation_config): temp 1.0, top_p 0.95, top_k 20
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 160}
    if with_tools:
        sampling["tools"] = TOOLS

    def run(**extra):
        out = chat(server, messages, **sampling, **extra)
        return out, (out["reasoning"], out["content"], out["usage"]["completion_tokens"])

    first, first_text = run()
    seed = first["lmk"]["seed"]
    assert isinstance(seed, int)
    replay, replay_text = run(seed=seed)
    again, again_text = run(seed=seed)
    other, other_text = run(seed=seed + 1)
    print(f"itest seed: {seed} tools={with_tools} draft={server.engine.draft_stats() is not None} "
          f"cached first/replay/again={first['usage']['prompt_tokens_details']['cached_tokens']}/"
          f"{replay['usage']['prompt_tokens_details']['cached_tokens']}/{again['usage']['prompt_tokens_details']['cached_tokens']} "
          f"drafted replay/again={replay['lmk'].get('draft_drafted')}/{again['lmk'].get('draft_drafted')} "
          f"cold-vs-restored identical={first_text == replay_text}")
    if first_text != replay_text:
        # not asserted: the first run read the prompt cold, the replays restored it; beyond rounding noise
        # at a near-tie this is a cache bug worth a look (backlog SPD-034)
        a, b = "".join(first_text[:2]), "".join(replay_text[:2])
        at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"itest seed: cold and restored runs part at char {at}: {a[max(0, at - 40):at + 40]!r} / {b[max(0, at - 40):at + 40]!r}")
    assert replay["lmk"]["seed"] == seed and other["lmk"]["seed"] == seed + 1
    assert replay_text == again_text
    assert other_text != replay_text


# design 2026-09-26-sampling-seed: draws are keyed by position, so with a draft on or off the same seed draws the
# same tokens as long as the logits agree. They agree exactly on short prompts (the model page: "on short prompts
# code and copy-editing matched exactly"); at long contexts the block verify rounds differently and a near-tie
# can flip (SLC-006/010), so this is asserted on a short prompt only.
def test_the_same_seed_draws_the_same_tokens_with_the_draft_on_and_off(server):
    if server.engine.draft_stats() is None:
        pytest.skip("no draft model loaded (LMK_ITEST_DRAFT)")
    import uuid

    engine = server.engine
    prompt = engine.chat_format().render(
        [{"role": "system", "content": f"Session {uuid.uuid4().hex}."},
         {"role": "user", "content": "Write a four-line poem about a lighthouse."}], None)
    sampling = {"temp": 1.0, "top_p": 0.95, "top_k": 20, "seed": 20260926}

    def run(speculative: bool):
        generation = engine.generate(prompt, max_tokens=120, request_id=f"itest-spec-{speculative}-{uuid.uuid4().hex[:6]}",
                                     on_prefill=lambda *a: True,
                                     sampling={**sampling, "speculative_decoding_toggle": speculative})
        return "".join(generation), generation.stats

    run(False)                    # reads the prompt once; both runs below restore the same prefix
    plain, plain_stats = run(False)
    spec, spec_stats = run(True)
    print(f"itest spec on/off: drafted {spec_stats.draft_drafted} accepted {spec_stats.draft_accepted} "
          f"identical={plain == spec}")
    assert spec_stats.draft_drafted and spec_stats.draft_drafted > 0
    assert spec == plain


# design 2026-09-26-context-check: the prefill scores how well the model predicts its own new prompt segment.
# (a) an intact restored conversation: the number, printed, and a loose sanity bound; (b) the same restore with
# the linear-attention (recurrent) states zeroed — the failure mode suspected in incident 000193 — must score
# clearly worse. That difference is the proof the check can see a lost context.
def test_the_context_check_sees_a_corrupted_restore(server):
    import uuid

    import mlx.core as mx
    import mlx_engine.model_kit.batched_vision.model_kit as vision_kit

    engine = server.engine
    fmt = engine.chat_format()
    names = [f"{w}-{i * 7919 % 1000:03d}" for i, w in enumerate(
        "amber basalt cobalt dune ember fjord garnet harbor indigo jasper kelp lumen marble nectar onyx pewter "
        "quartz russet sable tundra umber velvet willow xenon yarrow zephyr".split())]
    system = {"role": "system", "content": f"Session {uuid.uuid4().hex}. You are kitten, a coding agent."}
    history = [system,
               {"role": "user", "content": "Here are the release codenames, in order: " + ", ".join(names) +
                ". Keep them in mind. " + " ".join(f"Note {i}: codename {n} ships in week {i + 3}."
                                                   for i, n in enumerate(names))},
               {"role": "assistant", "content": "Understood — I have the codenames and their weeks."}]
    def turn(tag):
        # the tag leads the new message, so the second turn shares the first one's cache only up to the history
        return history + [{"role": "user", "content": f"{tag} Repeat every codename with its week, in the same "
                           "order, as 'codename: week' lines. Here they are once more for reference: " +
                           "; ".join(f"{n}: week {i + 3}" for i, n in enumerate(names))}]

    def run(messages):
        g = engine.generate(fmt.render(messages, None), max_tokens=4, request_id=f"itest-ctx-{uuid.uuid4().hex[:8]}",
                            on_prefill=lambda *a: True, sampling={"temp": 0.0})
        "".join(g)
        return g.stats

    cold = run(history)
    intact = run(turn("Request A."))           # restores the history, prefills the new turn
    zeroed = []

    def zero_recurrent(restored):
        for c in restored.prompt_cache:
            if type(c).__name__ == "ArraysCache":
                c.cache = [None if a is None else mx.zeros_like(a) for a in c.cache]
                zeroed.append(1)

    vision_kit.RESTORED_CACHE_HOOK = zero_recurrent
    try:
        corrupt = run(turn("Request B."))      # restores (about) the same point, recurrent state wiped
    finally:
        vision_kit.RESTORED_CACHE_HOOK = None
    for name, s in (("cold", cold), ("intact", intact), ("corrupt", corrupt)):
        c = s.context_check or {}
        per_1k = c.get("ms", 0) / max(1, c.get("scored_tokens") or 1) * 1000
        print(f"itest context check {name}: prompt={s.prompt_tokens} cached={s.cached_tokens} {c} "
              f"(~{per_1k:.0f} ms per 1k scored tokens)")

    a, b = intact.context_check, corrupt.context_check
    assert a["restore_source"] in ("hot", "disk") and a["restored_tokens"] > 0 and a["scored_tokens"] >= 32
    assert b["restored_tokens"] > 0 and b["scored_tokens"] >= 32
    assert zeroed, "the hook reached the recurrent layers"
    # loose sanity bound, not the band (that is to be measured): a copy task an intact model predicts well
    assert a["surprise_mean"] < 2.0
    assert b["surprise_mean"] > a["surprise_mean"] + 1.0
