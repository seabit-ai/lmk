from lmk.splitter import OutputSplitter

START, END = "<tool_call>", "</tool_call>"


def run(fragments, starts_in_reasoning):
    events = []
    s = OutputSplitter(START, END, starts_in_reasoning,
                       on_reasoning=lambda t: events.append(("r", t)),
                       on_text=lambda t: events.append(("t", t)),
                       on_tool_block=lambda b: events.append(("tool", b)))
    for f in fragments:
        s.write(f)
    s.close()
    merged = []
    for kind, val in events:  # coalesce neighbours so tests don't depend on fragment boundaries
        if merged and merged[-1][0] == kind and kind != "tool":
            merged[-1] = (kind, merged[-1][1] + val)
        else:
            merged.append((kind, val))
    return merged


# What exp02 recorded from the real model: the think-open tag lives in the PROMPT,
# so output starts inside reasoning; then two tool-call blocks.
def test_real_shape_reasoning_then_two_tool_blocks():
    out = run(["We need two calls.\n", "</think>", "\n\n", "<tool_call>\n<function=a>\n</function>\n</tool_call>",
               "\n<tool_call>\n<function=b>\n</function>\n</tool_call>"], starts_in_reasoning=True)
    assert out == [("r", "We need two calls.\n"),
                   ("tool", "\n<function=a>\n</function>\n"),
                   ("tool", "\n<function=b>\n</function>\n")]


def test_text_answer_after_reasoning_drops_the_blank_gap():
    assert run(["thinking", "</think>", "\n\nHere", " it is."], True) == [("r", "thinking"), ("t", "Here it is.")]


def test_markers_split_across_fragments_at_any_byte():
    out = run(["abc</th", "ink>\n\nhi <tool", "_call>X</tool_c", "all> bye"], True)
    # whitespace right after a tool block is dropped: the newline between two
    # parallel calls must not surface as an answer-text delta
    assert out == [("r", "abc"), ("t", "hi "), ("tool", "X"), ("t", "bye")]


def test_model_that_writes_its_own_think_open_tag():
    assert run(["<think>", "r", "</think>", "\n\nanswer"], False) == [("r", "r"), ("t", "answer")]


def test_no_reasoning_at_all():
    assert run(["plain ", "answer"], False) == [("t", "plain answer")]


def test_angle_bracket_that_is_not_a_marker_is_text():
    assert run(["a <b> c <tool", "box>"], False) == [("t", "a <b> c <toolbox>")]


# max_tokens hit while still thinking (Bruce's eval saw exactly this): all reasoning, no text.
def test_stream_ends_inside_reasoning():
    assert run(["still ", "thinking"], True) == [("r", "still thinking")]


# max_tokens hit inside a tool block: an unfinished call is never handed to the parser.
def test_stream_ends_inside_a_tool_block():
    assert run(["x</think>", "<tool_call>\n<function=a>"], True) == [("r", "x")]


def test_what_the_model_is_writing_right_now_is_read_off_the_markers():
    """`lmk status` shows this next to `decode`. The engine cannot tell: to it, it is all tokens."""
    s = OutputSplitter("<tool_call>", "</tool_call>", True, lambda t: None, lambda t: None, lambda b: None)
    assert s.part == "thinking"
    s.write("let me see</think>\n\nThe answer")
    assert s.part == "answering"
    s.write(" is 4.\n<tool_call>\n<function=file_read>")
    assert s.part == "tool call"
    s.write("</function>\n</tool_call>")
    assert s.part == "answering"
