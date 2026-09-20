"""One chat completion, as a stream of OpenAI-shaped chunks (design §5).

lmk's additions ride inside the shape so that stock OpenAI clients keep working:
  - cache hits:        usage.prompt_tokens_details.cached_tokens  (the standard field)
  - reasoning:         delta.reasoning_content
  - prefill progress:  a chunk with `choices: []` and an `lmk.prefill` object
"""
import json
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from lmk import log
from lmk.clock import get_current_clock
from lmk.engine import Engine
from lmk.splitter import OutputSplitter


@dataclass(frozen=True)
class CallerIdentity:
    """Who is calling and why (wish list WISH-007). All optional; absent is logged as absent."""
    purpose: Optional[str] = None
    ref_id: Optional[str] = None
    traceparent: Optional[str] = None


class ClientGone(Exception):
    """Raised by the chunk sink when the client disconnected: that IS the cancel signal."""


def run_chat(engine: Engine, body: dict, identity: CallerIdentity,
             emit: Callable[[dict], None], on_progress: Callable[[dict], None] = lambda _: None) -> dict:
    clock = get_current_clock()
    started = clock.mono_ms()
    fmt = engine.chat_format()
    tools = body.get("tools") or None
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
    prompt = fmt.render(body.get("messages") or [], tools)

    completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    base = {"id": completion_id, "created": clock.wall_ms() // 1000, "model": engine.loaded_model().id}
    state = {"cancelled": False, "first_ms": None, "calls": [], "text": [], "reasoning": []}

    def send(chunk: dict) -> None:
        if state["cancelled"]:
            return
        try:
            emit({**base, **chunk})
        except ClientGone:
            state["cancelled"] = True

    def delta(d: dict, finish: Optional[str] = None) -> None:
        if state["first_ms"] is None and ("content" in d or "reasoning_content" in d or "tool_calls" in d):
            state["first_ms"] = clock.mono_ms() - started
        send({"object": "chat.completion.chunk",
              "choices": [{"index": 0, "delta": d, "finish_reason": finish}]})

    def on_prefill(processed: int, total: int, cached: int) -> bool:
        progress = {"processed": processed, "total": total, "cached": cached}
        on_progress(progress)
        send({"object": "lmk.prefill", "choices": [], "lmk": {"prefill": progress}})
        return not state["cancelled"]

    def on_text(t: str) -> None:
        state["text"].append(t)
        delta({"content": t})

    def on_reasoning(t: str) -> None:
        state["reasoning"].append(t)
        delta({"reasoning_content": t})

    def on_tool_block(block: str) -> None:
        try:
            state["calls"].append(fmt.parse_tool_call(block, tools))
        except Exception as e:  # a malformed call must not vanish: surface what the model wrote
            log.warn("LmkToolCallParseFailed", "could not parse a tool-call block; passing it through as text",
                     refId=identity.ref_id, error=repr(e))
            on_text(f"{fmt.tool_call_start}{block}{fmt.tool_call_end}")

    request_id = identity.ref_id or completion_id
    generation = engine.generate(prompt, max_tokens=max_tokens, request_id=request_id, on_prefill=on_prefill)
    delta({"role": "assistant"})
    splitter = OutputSplitter(fmt.tool_call_start, fmt.tool_call_end, fmt.starts_in_reasoning(prompt),
                              on_reasoning, on_text, on_tool_block)
    for piece in generation:
        splitter.write(piece)
        if state["cancelled"]:
            generation.pieces.close()  # stops the engine's generator
            break
    splitter.close()

    tool_calls = [{"index": i, "id": "call_" + uuid.uuid4().hex[:16], "type": "function",
                   "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False)}}
                  for i, c in enumerate(state["calls"])]
    for call in tool_calls:
        delta({"tool_calls": [call]})
    stats = generation.stats
    if tool_calls:
        finish = "tool_calls"
    elif max_tokens and stats.completion_tokens >= max_tokens:
        finish = "length"
    else:
        finish = "stop"
    delta({}, finish)
    usage = {"prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.completion_tokens,
             "total_tokens": stats.prompt_tokens + stats.completion_tokens,
             "prompt_tokens_details": {"cached_tokens": stats.cached_tokens}}
    send({"object": "chat.completion.chunk", "choices": [], "usage": usage})

    total_ms = clock.mono_ms() - started
    log.info("LmkChatDone", "chat completion finished",
             purpose=identity.purpose, refId=identity.ref_id, traceparent=identity.traceparent,
             promptTokens=stats.prompt_tokens, cachedTokens=stats.cached_tokens,
             completionTokens=stats.completion_tokens, toolCalls=len(tool_calls),
             ttftMs=state["first_ms"], totalMs=total_ms, finishReason=finish, cancelled=state["cancelled"])
    return {"id": completion_id, "finish_reason": finish, "usage": usage, "tool_calls": tool_calls,
            "content": "".join(state["text"]), "reasoning_content": "".join(state["reasoning"]),
            "base": base, "cancelled": state["cancelled"]}
