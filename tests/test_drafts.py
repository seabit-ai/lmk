"""MTP heads for repo / path models (design 2026-10-20-draft-for-any-model)."""
import json

import pytest

from lmk import drafts
from lmk.drafts import DraftProblem

mx = pytest.importorskip("mlx.core")

HIDDEN = 64  # quantization groups are 64 wide, so the smallest honest tensor is 64 columns


def _model_dir(tmp_path, hidden=HIDDEN, vocab=100):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "quantization": {"group_size": 64, "bits": 4},
        "text_config": {"hidden_size": hidden, "vocab_size": vocab, "mtp_num_hidden_layers": 1,
                        "rope_parameters": {"type": "default", "rope_theta": 10000000}}}))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"language_model.x": "model.safetensors"}}))
    return d


def _sidecar(model_dir, bits=4, norm=0.0, layout_marker=True):
    """A tiny head in the layout of the community 4-bit conversions: optiq/mtp.safetensors, `mtp.` prefix,
    HF zero-centred norms, quantized projections."""
    w = mx.random.normal((8, HIDDEN))
    q, scales, biases = mx.quantize(w, group_size=64, bits=bits)
    side = model_dir / "optiq"
    side.mkdir()
    tensors = {"mtp.fc.weight": mx.zeros((HIDDEN, 2 * HIDDEN)).astype(mx.bfloat16),
               "mtp.norm.weight": mx.full((HIDDEN,), norm).astype(mx.bfloat16),
               "mtp.layers.0.self_attn.q_proj.weight": q, "mtp.layers.0.self_attn.q_proj.scales": scales,
               "mtp.layers.0.self_attn.q_proj.biases": biases}
    mx.save_safetensors(str(side / "mtp.safetensors"), tensors)
    if layout_marker:
        (side / "metadata.json").write_text('{"method": "optiq_mixed_precision"}')
    return side / "mtp.safetensors"


def test_a_model_without_a_head_is_refused_with_the_ways_out(tmp_path):
    model = _model_dir(tmp_path)
    with pytest.raises(DraftProblem) as e:
        drafts.own_head_draft(model, "rev1", tmp_path / "drafts")
    msg = str(e.value)
    assert "has no MTP head" in msg and "`official`" in msg and "speculative_decoding: false" in msg


def test_a_head_inside_the_main_weight_files_is_named_as_unsupported_not_as_missing(tmp_path):
    model = _model_dir(tmp_path)
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"mtp.fc.weight": "model-00004.safetensors"}}))
    with pytest.raises(DraftProblem, match="inside its main weight files"):
        drafts.own_head_draft(model, "rev1", tmp_path / "drafts")


def test_a_sidecar_in_a_layout_we_have_not_checked_is_refused(tmp_path):
    model = _model_dir(tmp_path)
    _sidecar(model, layout_marker=False)
    with pytest.raises(DraftProblem, match="layout lmk has not checked"):
        drafts.own_head_draft(model, "rev1", tmp_path / "drafts")


def test_the_conversion_makes_a_folder_the_engine_can_load(tmp_path):
    model = _model_dir(tmp_path)
    _sidecar(model, norm=0.5)
    out = drafts.own_head_draft(model, "rev1", tmp_path / "drafts")
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "qwen3_5_mtp" and cfg["block_size"] == 3 and cfg["tie_word_embeddings"] is False
    assert cfg["text_config"]["hidden_size"] == HIDDEN
    assert cfg["text_config"]["rope_parameters"] == {"rope_type": "default", "rope_theta": 10000000}  # the draft's spelling
    assert cfg["quantization"]["layers.0.self_attn.q_proj"] == {"bits": 4, "group_size": 64}
    t = mx.load(str(out / "model.safetensors"))
    assert "fc.weight" in t and not any(k.startswith("mtp.") for k in t)
    assert mx.allclose(t["norm.weight"].astype(mx.float32), mx.array(1.5)).item()  # zero-centred norms + 1
    assert t["fc.weight"].dtype == mx.bfloat16 and mx.all(t["fc.weight"] == 0).item()  # other tensors untouched


def test_eight_bit_layers_are_told_apart_from_four_bit_by_shape(tmp_path):
    model = _model_dir(tmp_path)
    _sidecar(model, bits=8)
    out = drafts.own_head_draft(model, "rev1", tmp_path / "drafts")
    assert json.loads((out / "config.json").read_text())["quantization"]["layers.0.self_attn.q_proj"]["bits"] == 8


def test_converting_twice_reuses_the_folder_and_a_new_revision_makes_a_new_one(tmp_path):
    model = _model_dir(tmp_path)
    _sidecar(model)
    first = drafts.own_head_draft(model, "rev1", tmp_path / "drafts")
    stamp = (first / "model.safetensors").stat().st_mtime_ns
    assert drafts.own_head_draft(model, "rev1", tmp_path / "drafts") == first
    assert (first / "model.safetensors").stat().st_mtime_ns == stamp
    assert drafts.own_head_draft(model, "rev2", tmp_path / "drafts") != first


def test_a_draft_that_does_not_fit_is_refused_with_both_numbers(tmp_path):
    model = _model_dir(tmp_path, hidden=HIDDEN)
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "qwen3_5_mtp", "text_config": {"hidden_size": 5120, "vocab_size": 248320}}))
    problem = drafts.incompatible(other, model)
    assert "hidden_size is 5120" in problem and f"model's is {HIDDEN}" in problem
    (other / "config.json").write_text(json.dumps({"model_type": "qwen3_5_mtp", "text_config": {"hidden_size": HIDDEN, "vocab_size": 7}}))
    assert "vocab_size is 7" in drafts.incompatible(other, model)
    (other / "config.json").write_text(json.dumps({"model_type": "qwen3_5_mtp", "text_config": {"hidden_size": HIDDEN, "vocab_size": 100}}))
    assert drafts.incompatible(other, model) is None


def test_a_dflash_drafter_is_not_an_mtp_head(tmp_path):
    model = _model_dir(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"architectures": ["DFlash2DraftModel"], "dflash_config": {}, "hidden_size": HIDDEN}))
    assert "DFlash" in drafts.incompatible(other, model)


def test_the_weights_setting_tells_a_folder_from_a_repo_by_its_first_characters():
    assert [drafts.is_folder(v) for v in ("./official", "../d", "/abs/d", "~/d", "org/name", "model", "official")] == \
        [True, True, True, True, False, False, False]
