from lmk.stopmatch import StopMatcher


def collect(stops, pieces):
    out = []
    m = StopMatcher(stops, out.append)
    hit = False
    for p in pieces:
        if m.write(p):
            hit = True
            break
    if not hit:
        m.close()
    return "".join(out), hit


def test_text_before_the_stop_string_is_emitted_and_the_string_itself_is_not():
    assert collect(["END"], ["Hello ", "world END and more"]) == ("Hello world ", True)


def test_a_stop_string_split_across_pieces_is_still_found():
    assert collect(["\n\n"], ["line one\n", "\nline two"]) == ("line one", True)
    assert collect(["STOP"], ["S", "T", "O", "P!"]) == ("", True)


def test_a_partial_match_is_held_back_only_until_it_is_ruled_out():
    out = []
    m = StopMatcher(["STOP"], out.append)
    m.write("say ST")
    assert "".join(out) == "say "       # "ST" might be the start of STOP: held
    m.write("ART")
    assert "".join(out) == "say START"  # ruled out: released
    m.close()
    assert "".join(out) == "say START"


def test_close_flushes_a_held_tail_when_generation_ends_without_a_stop():
    assert collect(["STOP"], ["until the ST"]) == ("until the ST", False)


def test_the_earliest_of_several_stop_strings_wins():
    assert collect(["<b>", "</b>"], ["x </b> y <b> z"]) == ("x ", True)


def test_no_stop_strings_is_a_pass_through():
    assert collect([], ["a", "b"]) == ("ab", False)
