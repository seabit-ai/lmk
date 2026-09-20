from lmk.chatformat import _for_template


def test_tool_call_arguments_become_a_mapping_for_the_template():
    out = _for_template({"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "f", "arguments": '{"path": "a", "n": 2}'}}]})
    assert out["content"] == ""
    assert out["tool_calls"][0]["function"]["arguments"] == {"path": "a", "n": 2}


def test_unparseable_arguments_do_not_break_rendering():
    out = _for_template({"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]})
    assert out["tool_calls"][0]["function"]["arguments"] == {}
