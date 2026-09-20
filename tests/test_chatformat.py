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
