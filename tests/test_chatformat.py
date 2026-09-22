from lmk.chatformat import _for_template


def test_tool_call_arguments_become_a_mapping_for_the_template():
    out = _for_template({"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "f", "arguments": '{"path": "a", "n": 2}'}}]})
    assert out["content"] == ""
    assert out["tool_calls"][0]["function"]["arguments"] == {"path": "a", "n": 2}


def test_unparseable_arguments_do_not_break_rendering():
    out = _for_template({"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]})
    assert out["tool_calls"][0]["function"]["arguments"] == {}


import pytest

from lmk.chatformat import ImageInputError, split_images

PNG = "data:image/png;base64,AAAA"
JPG = "data:image/jpeg;base64,BBBB"


def test_images_become_template_placeholders_and_an_ordered_base64_list():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "compare"},
                                     {"type": "image_url", "image_url": {"url": PNG}},
                                     {"type": "image_url", "image_url": {"url": JPG}}]},
        {"role": "assistant", "content": "ok"},
    ]
    for_template, images = split_images(messages)
    assert images == ["AAAA", "BBBB"]
    assert for_template[1]["content"] == [{"type": "text", "text": "compare"}, {"type": "image"}, {"type": "image"}]
    assert for_template[0] == messages[0] and for_template[2] == messages[2]
    assert messages[1]["content"][1]["type"] == "image_url", "the caller's messages are not mutated"


def test_text_only_conversations_pass_through_untouched():
    messages = [{"role": "user", "content": "hi"}]
    assert split_images(messages) == (messages, [])


# The server never fetches a URL on a client's say-so; bytes travel inline.
@pytest.mark.parametrize("url", ["https://example.com/cat.png", "file:///etc/passwd", "data:image/png,notbase64"])
def test_only_inline_base64_data_urls_are_accepted(url):
    with pytest.raises(ImageInputError):
        split_images([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}])


def test_template_kwargs_reach_the_chat_template_on_every_render():
    from lmk.chatformat import TemplateChatFormat

    class Tok:
        chat_template = ""
        calls = []

        def apply_chat_template(self, messages, **kw):
            self.calls.append(kw)
            return "P"

    tok = Tok()
    fmt = TemplateChatFormat(tok, {"enable_thinking": False, "reasoning_effort": "low"})
    fmt.render([{"role": "user", "content": "hi"}], None)
    assert tok.calls[0]["enable_thinking"] is False and tok.calls[0]["reasoning_effort"] == "low"
    assert TemplateChatFormat(Tok()).render([], None) == "P"     # no kwargs: the template's defaults


# --- dialects: the per-family knowledge, chosen from the template text

def test_the_dialect_is_read_off_the_chat_template():
    from lmk.chatformat import GEMMA4, GEMMA4_LEGACY_TOOLS, PLAIN, QWEN, dialect_for_template

    assert dialect_for_template("... {{ '<think>\n' }} ...") is QWEN
    assert dialect_for_template("... {{- '<|channel>thought\n<channel|>' -}} ... tc.get('id') == follow.get('tool_call_id')") is GEMMA4
    assert dialect_for_template("... {{- '<|channel>thought\n<channel|>' -}} ... message['tool_responses'] ...") is GEMMA4_LEGACY_TOOLS
    assert dialect_for_template("{% for m in messages %}{{ m.content }}{% endfor %}") is PLAIN
    assert dialect_for_template(None) is PLAIN


def test_qwen_thinks_by_default_and_the_prompt_can_end_inside_the_think_block():
    from lmk.chatformat import QWEN

    assert QWEN.thinking_default is True and QWEN.prompt_decides_thinking is True and QWEN.think_open == "<think>" and QWEN.think_close == "</think>"
    assert QWEN.starts_in_reasoning("...<|im_start|>assistant\n<think>\n") is True
    assert QWEN.starts_in_reasoning("...<|im_start|>assistant\n<think>\n\n</think>\n\n") is False


def test_gemma_does_not_think_unless_asked_and_opens_its_own_thought_channel():
    from lmk.chatformat import GEMMA4

    assert GEMMA4.thinking_default is False and GEMMA4.prompt_decides_thinking is False
    assert (GEMMA4.think_open, GEMMA4.think_close) == ("<|channel>thought\n", "<channel|>")
    # thinking on: the prompt ends with the model turn and the model writes the channel itself
    assert GEMMA4.starts_in_reasoning("...<turn|>\n<|turn>model\n") is False
    # thinking off: the template closes an empty thought for the model
    assert GEMMA4.starts_in_reasoning("...<|turn>model\n<|channel>thought\n<channel|>") is False


def test_the_current_gemma_template_takes_openai_tool_results_as_they_are():
    from lmk.chatformat import GEMMA4

    wire = [{"role": "tool", "tool_call_id": "c", "content": "x"}]
    assert GEMMA4.for_template(wire) == [{"role": "tool", "tool_call_id": "c", "content": "x"}]


def test_the_earlier_gemma_template_needs_tool_responses_named_after_the_call():
    from lmk.chatformat import GEMMA4_LEGACY_TOOLS as GEMMA4

    wire = [{"role": "user", "content": "read it"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "file_read", "arguments": "{\"path\": \"notes.md\"}"}},
                {"id": "call_2", "type": "function", "function": {"name": "weather", "arguments": "{\"city\": \"Oslo\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "# notes\n- x"},
            {"role": "tool", "tool_call_id": "call_2", "content": "{\"temp_c\": 7, \"sky\": \"grey\"}"}]
    out = GEMMA4.for_template(wire)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"path": "notes.md"}   # a mapping, as for Qwen
    assert out[2] == {"role": "tool", "tool_responses": [{"name": "file_read", "response": "# notes\n- x"}]}
    assert out[3] == {"role": "tool", "tool_responses": [{"name": "weather", "response": {"temp_c": 7, "sky": "grey"}}]}


def test_a_tool_result_whose_call_is_unknown_keeps_going_with_an_unknown_name():
    from lmk.chatformat import GEMMA4_LEGACY_TOOLS as GEMMA4

    out = GEMMA4.for_template([{"role": "tool", "tool_call_id": "nope", "content": "x"}])
    assert out[0]["tool_responses"] == [{"name": "unknown", "response": "x"}]


def test_qwen_leaves_tool_results_in_openai_shape():
    from lmk.chatformat import QWEN

    wire = [{"role": "tool", "tool_call_id": "c", "content": "x"}]
    assert QWEN.for_template(wire) == [{"role": "tool", "tool_call_id": "c", "content": "x"}]
