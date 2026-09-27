import json

import pytest

from lmk.sampling import SamplingError, SeedSource, model_defaults, parse_sampling


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


@pytest.mark.parametrize("seed", [0, 42, 2**63, 2**64 - 1])
def test_seed_reaches_the_engine_as_given(seed):
    sampling, ignored = parse_sampling({"seed": seed}, defaults={})
    assert sampling == {"seed": seed} and ignored == []


def test_a_drawn_seed_fits_in_31_bits():
    assert all(0 <= SeedSource().draw() < 2**31 for _ in range(200))


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
    ({"seed": 1.5}, "seed", "an integer from 0 to 18446744073709551615"),
    ({"seed": "42"}, "seed", "an integer from 0 to 18446744073709551615"),
    ({"seed": True}, "seed", "an integer from 0 to 18446744073709551615"),
    ({"seed": -1}, "seed", "an integer from 0 to 18446744073709551615"),
    ({"seed": 2**64}, "seed", "an integer from 0 to 18446744073709551615"),
])
def test_out_of_range_values_name_the_field_and_the_rule(body, param, words):
    with pytest.raises(SamplingError) as e:
        parse_sampling(body, defaults={})
    assert e.value.param == param and words in str(e.value)


def test_booleans_are_not_numbers():
    with pytest.raises(SamplingError):
        parse_sampling({"temperature": True}, defaults={})
