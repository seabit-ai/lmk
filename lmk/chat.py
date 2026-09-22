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
from lmk.chatformat import ImageInputError, split_images
from lmk.engine import Engine, Preflight
from lmk.sampling import parse_sampling
from lmk.splitter import OutputSplitter
from lmk.stopmatch import StopMatcher


@dataclass(frozen=True)
class CallerIdentity:
    """Who is calling and why (wish list WISH-007). All optional; absent is logged as absent."""
    purpose: Optional[str] = None
    ref_id: Optional[str] = None
    traceparent: Optional[str] = None


class ClientGone(Exception):
    """Raised by the chunk sink when the client disconnected: that IS the cancel signal."""


def prepare_messages(engine: Engine, body: dict) -> tuple[list[dict], list[str]]:
    """Everything that can reject a request, done BEFORE any response byte is
    sent (a streamed reply cannot turn into a 400 halfway)."""
    messages, images = split_images(body.get("messages") or [])
    if images and "image" not in engine.input_modalities():
        raise ImageInputError("the resident model does not take images")
    return messages, images


@dataclass
class PreparedChat:
    """The request rendered and measured — everything the admission queue needs to know,
    done once and handed on to run_chat / run_warmup."""
    prompt: str
    images: list
    tools: Optional[list]
    max_tokens: Optional[int]
    preflight: Preflight
    sampling: dict            # engine kwargs (lmk.sampling); never stop_strings — see stop_strings below
    stop_strings: list[str]   # OpenAI `stop`, matched by lmk on the answer part only (lmk.stopmatch)
    ignored_params: list[str]  # request fields lmk understood but cannot honour (logged, not refused)

    def tokens_needed(self, context_length: int) -> int:
        """Its share of the KV memory: the prompt plus what it may write, capped by the window."""
        return min(self.preflight.prompt_tokens + (self.max_tokens or context_length), context_length)


def prepare_chat(engine: Engine, body: dict, warmup: bool = False) -> PreparedChat:
    messages, images = prepare_messages(engine, body)
    tools = body.get("tools") or None
    if warmup:
        messages = list(messages)
        if not messages or messages[-1].get("role") != "user":
            # chat templates want a user turn to close on; keep it tiny so the fork
            # point stays inside the last cache block
            messages.append({"role": "user", "content": "."})
    prompt = engine.chat_format().render(messages, tools)
    max_tokens = 1 if warmup else (body.get("max_tokens") or body.get("max_completion_tokens"))
    sampling, ignored = parse_sampling(body, engine.sampling_defaults())
    stop_strings = sampling.pop("stop_strings", [])
    return PreparedChat(prompt=prompt, images=images, tools=tools, max_tokens=max_tokens,
                        preflight=engine.preflight(prompt, images), sampling=sampling, stop_strings=stop_strings,
                        ignored_params=ignored)


def run_chat(engine: Engine, body: dict, identity: CallerIdentity,
             emit: Callable[[dict], None], on_progress: Callable[[dict], None] = lambda _: None,
             prepared: Optional[PreparedChat] = None) -> dict:
    clock = get_current_clock()
    started = clock.mono_ms()
    fmt = engine.chat_format()
    prepared = prepared or prepare_chat(engine, body)
    tools, max_tokens, images, prompt = prepared.tools, prepared.max_tokens, prepared.images, prepared.prompt

    completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    base = {"id": completion_id, "created": clock.wall_ms() // 1000, "model": engine.loaded_model().id}
    state = {"cancelled": False, "stopped": False, "first_ms": None, "calls": [], "text": [], "reasoning": [],
             "generate_called_ms": None, "restore_ms": None}

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
        if state["restore_ms"] is None and state["generate_called_ms"] is not None:
            # from handing the request to the engine until it starts reading the uncached part:
            # the cached part coming back from disk (plus the wait for the engine's scheduler)
            state["restore_ms"] = clock.mono_ms() - state["generate_called_ms"]
        progress = {"processed": processed, "total": total, "cached": cached}
        on_progress({"prefill": progress})
        send({"object": "lmk.prefill", "choices": [], "lmk": {"prefill": progress}})
        return not state["cancelled"]

    def emit_text(t: str) -> None:
        state["text"].append(t)
        delta({"content": t})

    stop_matcher = StopMatcher(prepared.stop_strings, emit_text)

    def on_text(t: str) -> None:
        if not state["stopped"] and stop_matcher.write(t):
            state["stopped"] = True

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
    state["generate_called_ms"] = clock.mono_ms()
    if prepared.ignored_params:
        log.warn("LmkParamIgnored", "request fields the engine cannot honour", purpose=identity.purpose,
                 refId=identity.ref_id, params=prepared.ignored_params)
    generation = engine.generate(prompt, max_tokens=max_tokens, request_id=request_id, on_prefill=on_prefill,
                                 images_b64=images, tokens=prepared.preflight.tokens, sampling=prepared.sampling)
    delta({"role": "assistant"})
    splitter = OutputSplitter(fmt.tool_call_start, fmt.tool_call_end, fmt.starts_in_reasoning(prompt),
                              on_reasoning, on_text, on_tool_block)
    for piece in generation:
        splitter.write(piece)
        on_progress({"decode": {"part": splitter.part, "completion_tokens": generation.stats.completion_tokens}})
        if state["cancelled"] or state["stopped"]:
            generation.pieces.close()  # stops the engine's generator
            break
    splitter.close()
    if not state["stopped"]:
        stop_matcher.close()

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
    # lmk's own timing next to the standard usage: how long the cached part took to come back
    # from disk (what a client cannot see from outside; `lmk bench` reads it)
    send({"object": "chat.completion.chunk", "choices": [], "usage": usage,
          "lmk": {"restore_ms": state["restore_ms"], "first_token_ms": state["first_ms"]}})

    total_ms = clock.mono_ms() - started
    log.info("LmkChatDone", "chat completion finished",
             purpose=identity.purpose, refId=identity.ref_id, traceparent=identity.traceparent,
             promptTokens=stats.prompt_tokens, cachedTokens=stats.cached_tokens,
             # what the queue believed before admission, next to what the engine then found: if these
             # drift apart, rule 2 of the queue is judging "long" and "short" wrongly
             uncachedEstimate=prepared.preflight.uncached_tokens,
             uncachedActual=stats.prompt_tokens - stats.cached_tokens,
             completionTokens=stats.completion_tokens, toolCalls=len(tool_calls),
             restoreMs=state["restore_ms"], ttftMs=state["first_ms"], totalMs=total_ms, finishReason=finish, cancelled=state["cancelled"],
             sampling=prepared.sampling, stop=prepared.stop_strings or None)
    return {"id": completion_id, "finish_reason": finish, "usage": usage, "tool_calls": tool_calls,
            "content": "".join(state["text"]), "reasoning_content": "".join(state["reasoning"]),
            "base": base, "cancelled": state["cancelled"]}


def run_warmup(engine: Engine, body: dict, identity: CallerIdentity,
               prepared: Optional[PreparedChat] = None) -> dict:
    """Prefill a prefix into the cache without generating an answer (wish list
    WISH-019). The caller sends the part it wants warm — typically system +
    tools. A later request that starts the same way restores from the largest
    checkpointed 256-token boundary inside the shared prefix (research LMK-002).
    """
    clock = get_current_clock()
    started = clock.mono_ms()
    prepared = prepared or prepare_chat(engine, body, warmup=True)
    request_id = identity.ref_id or "warmup-" + uuid.uuid4().hex[:16]
    generation = engine.generate(prepared.prompt, max_tokens=1, request_id=request_id, on_prefill=lambda *_: True,
                                 images_b64=prepared.images, tokens=prepared.preflight.tokens)
    for _ in generation:
        pass
    stats = generation.stats
    total_ms = clock.mono_ms() - started
    log.info("LmkWarmupDone", "prefix warmed", purpose=identity.purpose or "warmup", refId=identity.ref_id,
             traceparent=identity.traceparent, promptTokens=stats.prompt_tokens,
             cachedTokens=stats.cached_tokens, totalMs=total_ms)
    return {"prompt_tokens": stats.prompt_tokens, "cached_tokens": stats.cached_tokens, "total_ms": total_ms}
