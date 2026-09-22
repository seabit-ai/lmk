import json

import pytest

from lmk.sampling import SamplingError, model_defaults, parse_sampling


def test_defaults_come_from_the_models_generation_config(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "eos_token_id": [1, 2]}))
    assert model_defaults(tmp_path) == {"temp": 1.0, "top_k": 20, "top_p": 0.95}


def test_a_model_that_asks_for_greedy_decoding_gets_it(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": False, "temperature": 0.7}))
    assert model_defaults(tmp_path) == {"temp": 0.0}


def test_no_generation_config_means_no_defaults(tmp_path):
    assert model_defaults(tmp_path) == {}
    (tmp_path / "generation_config.json").write_text("not json")
    assert model_defaults(tmp_path) == {}


def test_request_values_override_the_defaults_under_the_engines_names():
    sampling, ignored = parse_sampling(
        {"temperature": 0.2, "top_p": 0.5, "top_k": 40, "min_p": 0.05, "repetition_penalty": 1.1, "stop": "\n\n"},
        defaults={"temp": 1.0, "top_p": 0.95, "top_k": 20})
    assert sampling == {"temp": 0.2, "top_p": 0.5, "top_k": 40, "min_p": 0.05, "repetition_penalty": 1.1,
                        "stop_strings": ["\n\n"]}
    assert ignored == []


def test_absent_and_null_fields_leave_the_defaults_alone():
    sampling, _ = parse_sampling({"temperature": None, "stop": None, "stop_sequences": ["x"]},
                                 defaults={"temp": 1.0})
    assert sampling == {"temp": 1.0}          # stop_sequences is not an OpenAI field: ignored, not an error


def test_seed_is_reported_as_ignored_not_silently_dropped():
    sampling, ignored = parse_sampling({"seed": 42}, defaults={})
    assert sampling == {} and ignored == ["seed"]


@pytest.mark.parametrize("body, param, words", [
    ({"temperature": 2.5}, "temperature", "between 0 and 2"),
    ({"temperature": "hot"}, "temperature", "a number"),
    ({"top_p": 0}, "top_p", "between 0 (exclusive) and 1"),
    ({"top_p": 1.5}, "top_p", "between 0 (exclusive) and 1"),
    ({"top_k": -1}, "top_k", "an integer of 0 or more"),
    ({"top_k": 2.5}, "top_k", "an integer of 0 or more"),
    ({"min_p": 1.5}, "min_p", "between 0 and 1"),
    ({"repetition_penalty": 0}, "repetition_penalty", "greater than 0"),
    ({"stop": ""}, "stop", "non-empty"),
    ({"stop": ["a", ""]}, "stop", "non-empty"),
    ({"stop": ["a", "b", "c", "d", "e"]}, "stop", "at most 4"),
    ({"stop": 7}, "stop", "a string or a list of strings"),
])
def test_out_of_range_values_name_the_field_and_the_rule(body, param, words):
    with pytest.raises(SamplingError) as e:
        parse_sampling(body, defaults={})
    assert e.value.param == param and words in str(e.value)


def test_booleans_are_not_numbers():
    with pytest.raises(SamplingError):
        parse_sampling({"temperature": True}, defaults={})
