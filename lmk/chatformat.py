"""The per-model-family seam (design §4): how messages + tools become a prompt,
and how the model's tool-call text becomes structured data. Both halves come
from open source — the model's own chat template and mlx-lm's tool parsers —
selected once when the model loads. Nothing outside this file knows a format.
"""
import importlib
import json
from typing import Any, Optional, Protocol


class ChatFormat(Protocol):
    tool_call_start: Optional[str]
    tool_call_end: Optional[str]

    def render(self, messages: list[dict], tools: Optional[list[dict]]) -> str: ...
    def starts_in_reasoning(self, prompt_text: str) -> bool: ...
    def parse_tool_call(self, block: str, tools: Optional[list[dict]]) -> dict: ...


class TemplateChatFormat:
    def __init__(self, tokenizer: Any):
        from mlx_lm.tokenizer_utils import _infer_tool_parser

        self._tokenizer = tokenizer
        parser_type = _infer_tool_parser(getattr(tokenizer, "chat_template", None))
        self._parser = importlib.import_module(f"mlx_lm.tool_parsers.{parser_type}") if parser_type else None
        self.tool_call_start = getattr(self._parser, "tool_call_start", None)
        self.tool_call_end = getattr(self._parser, "tool_call_end", None)
        self.parser_type = parser_type

    def render(self, messages, tools):
        return self._tokenizer.apply_chat_template(
            [_for_template(m) for m in messages], tools=tools or None,
            tokenize=False, add_generation_prompt=True)

    def starts_in_reasoning(self, prompt_text):
        # Some templates (Qwen3.5) put the think-open tag in the generation
        # prompt, so the output begins mid-reasoning (research LMK-003).
        return prompt_text.rstrip().endswith("<think>")

    def parse_tool_call(self, block, tools):
        if self._parser is None:
            raise ValueError("this model's chat template declares no tool-call format")
        return self._parser.parse_tool_call(block.strip(), tools)


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
