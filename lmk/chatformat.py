"""The per-model-family seam (design §4): how messages + tools become a prompt,
and how the model's tool-call text becomes structured data. Both halves come
from open source — the model's own chat template and mlx-lm's tool parsers —
selected once when the model loads. What differs between families beyond that
(thinking markers, whether thinking is on unless asked, the shape tool results
take) is a `Dialect`, chosen from the template text. Nothing outside this file
knows a format.
"""
import importlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from lmk import log
from lmk.splitter import Markers


class ChatFormat(Protocol):
    tool_call_start: Optional[str]
    tool_call_end: Optional[str]
    think_open: Optional[str]
    think_close: Optional[str]

    def render(self, messages: list[dict], tools: Optional[list[dict]]) -> str: ...
    def starts_in_reasoning(self, prompt_text: str) -> bool: ...
    def parse_tool_call(self, block: str, tools: Optional[list[dict]]) -> dict: ...


@dataclass(frozen=True)
class Dialect:
    """What a model family does that its chat template and tool parser do not say for us."""
    name: str
    think_open: Optional[str]
    think_close: Optional[str]
    thinking_default: bool                       # with no enable_thinking passed, does the template turn thinking on?
    prompt_decides_thinking: bool                # Qwen: the prompt opens or closes the think block, so on = every turn
                                                 # thinks and off = none does. Gemma: the prompt only hints; the model
                                                 # decides per turn either way (exp05 F1, F3).
    starts_in_reasoning: Callable[[str], bool]   # does the rendered prompt end inside a think block?
    for_template: Callable[[list[dict]], list[dict]]   # OpenAI wire messages → what this template iterates over


def _qwen_starts_in_reasoning(prompt_text: str) -> bool:
    # Qwen templates put the think-open tag in the generation prompt, so the output
    # begins mid-reasoning (research LMK-003); with thinking off they close it first
    return prompt_text.rstrip().endswith("<think>")


def _gemma_for_template(messages: list[dict]) -> list[dict]:
    """Gemma's template renders tool results from `tool_responses: [{name, response}]`
    on the message, named after the function — OpenAI's wire carries only the call id.
    A JSON-object result becomes a mapping (the template renders key:value); any other
    text goes through as it is."""
    names_by_call_id: dict[str, str] = {}
    out = []
    for message in messages:
        if message.get("role") != "tool":
            converted = _for_template(message)
            for call in converted.get("tool_calls") or []:
                if call.get("id"):
                    names_by_call_id[call["id"]] = (call.get("function") or {}).get("name", "unknown")
            out.append(converted)
            continue
        content = message.get("content")
        response: Any = content if content is not None else ""
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    response = parsed
            except ValueError:
                pass
        out.append({"role": "tool", "tool_responses": [
            {"name": names_by_call_id.get(message.get("tool_call_id") or "", "unknown"), "response": response}]})
    return out


QWEN = Dialect(name="qwen", think_open="<think>", think_close="</think>", thinking_default=True, prompt_decides_thinking=True,
               starts_in_reasoning=_qwen_starts_in_reasoning,
               for_template=lambda messages: [_for_template(m) for m in messages])

# Gemma 4: thinking is off unless enable_thinking is passed; when on, the model opens its own
# thought channel (the generation prompt is just the model turn); when off, the template closes an
# empty channel for it — and the model may still open one (exp05 F3). The model's answer follows a
# tool result in that same turn (the template adds no model turn after one).
# Tool results: the current mlx-community template (2026-07, commit 0d77464) takes OpenAI's
# `role: tool` + `tool_call_id` and resolves the function name itself; the earlier template only
# knew `tool_responses: [{name, response}]` ("legacy" in the new one's own words). Which one a
# model ships is read off the template.
GEMMA4 = Dialect(name="gemma4", think_open="<|channel>thought\n", think_close="<channel|>", thinking_default=False,
                 prompt_decides_thinking=False, starts_in_reasoning=lambda prompt: False,
                 for_template=lambda messages: [_for_template(m) for m in messages])
GEMMA4_LEGACY_TOOLS = Dialect(name="gemma4-legacy-tools", think_open=GEMMA4.think_open, think_close=GEMMA4.think_close,
                              thinking_default=False, prompt_decides_thinking=False,
                              starts_in_reasoning=lambda prompt: False, for_template=_gemma_for_template)

# A template we do not know: no thinking split (everything is answer text or a tool call),
# messages passed as OpenAI shapes them.
PLAIN = Dialect(name="plain", think_open=None, think_close=None, thinking_default=False, prompt_decides_thinking=False,
                starts_in_reasoning=lambda prompt: False,
                for_template=lambda messages: [_for_template(m) for m in messages])


def dialect_for_template(chat_template: Optional[str]) -> Dialect:
    text = chat_template or ""
    if "<|channel>thought" in text:
        return GEMMA4 if "tool_call_id" in text else GEMMA4_LEGACY_TOOLS
    if "<think>" in text:
        return QWEN
    return PLAIN


class TemplateChatFormat:
    def __init__(self, tokenizer: Any, template_kwargs: Optional[dict] = None):
        """template_kwargs: server-level constants handed to the chat template on every render —
        `enable_thinking`, `reasoning_effort`. The template renders them at the very start of the
        prompt, so they must never vary per request or the whole conversation goes cold
        (kitten design 2026-09-19-llm-call-flow-control §6.2)."""
        from mlx_lm.tokenizer_utils import _infer_tool_parser

        self._tokenizer = tokenizer
        self._template_kwargs = dict(template_kwargs or {})
        template = getattr(tokenizer, "chat_template", None)
        parser_type = _infer_tool_parser(template)
        self._parser = importlib.import_module(f"mlx_lm.tool_parsers.{parser_type}") if parser_type else None
        self.tool_call_start = getattr(self._parser, "tool_call_start", None)
        self.tool_call_end = getattr(self._parser, "tool_call_end", None)
        self.parser_type = parser_type
        self.dialect = dialect_for_template(template)
        self.think_open, self.think_close = self.dialect.think_open, self.dialect.think_close
        if self.dialect is PLAIN:
            log.warn("LmkTemplateDialectUnknown", "this chat template is not one lmk knows: no thinking will be "
                     "split out of the answer; tool calls follow mlx-lm's parser if any", toolParser=parser_type)

    def render(self, messages, tools):
        return self._tokenizer.apply_chat_template(
            self.dialect.for_template(messages), tools=tools or None,
            tokenize=False, add_generation_prompt=True, **self._template_kwargs)

    def starts_in_reasoning(self, prompt_text):
        return self.dialect.starts_in_reasoning(prompt_text)

    def markers(self) -> Markers:
        return Markers(self.tool_call_start, self.tool_call_end, self.think_open, self.think_close)

    def parse_tool_call(self, block, tools):
        if self._parser is None:
            raise ValueError("this model's chat template declares no tool-call format")
        return self._parser.parse_tool_call(block.strip(), tools)


class ImageInputError(ValueError):
    pass


def split_images(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """OpenAI image parts → what the engine wants: `{"type": "image"}` placeholders
    in the conversation (the template turns them into vision tokens) plus the
    images as base64, in order of appearance. Only inline `data:` URLs — the
    server never fetches a URL on a client's say-so."""
    images: list[str] = []
    out = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = []
        for part in content:
            if part.get("type") != "image_url":
                parts.append(part)
                continue
            url = (part.get("image_url") or {}).get("url") or ""
            head, sep, payload = url.partition(";base64,")
            if not (head.startswith("data:image/") and sep and payload):
                raise ImageInputError("images must be inline data:image/...;base64,... URLs")
            images.append(payload)
            parts.append({"type": "image"})
        out.append({**message, "content": parts})
    return out, images


def _for_template(message: dict) -> dict:
    """OpenAI wire → what chat templates iterate over: tool-call arguments are a
    JSON string on the wire but templates expect a mapping."""
    out = dict(message)
    if out.get("content") is None:
        out["content"] = ""
    calls = out.get("tool_calls")
    if calls:
        converted = []
        for call in calls:
            fn = dict(call.get("function") or {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    fn["arguments"] = {}
            converted.append({**call, "function": fn})
        out["tool_calls"] = converted
    return out
