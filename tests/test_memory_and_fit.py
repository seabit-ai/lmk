from lmk.memory import MemoryReading, SystemMemory
from lmk.modelfit import RESERVE_BYTES, weights_bytes, why_it_does_not_fit

GB = 1024**3


def test_this_machine_reports_plausible_numbers():
    r = SystemMemory().read()
    assert r.pressure in ("normal", "warning", "critical")
    assert 0 <= r.free_percent <= 100
    assert r.total_bytes >= 8 * GB


def test_only_critical_is_critical():
    assert MemoryReading("critical", 3, 96 * GB).critical
    assert not MemoryReading("warning", 12, 96 * GB).critical


def model(tmp_path, *shard_sizes):
    for i, size in enumerate(shard_sizes):
        with open(tmp_path / f"model-{i}.safetensors", "wb") as f:
            f.truncate(size)  # sparse: a "70 GB" shard costs no disk
    (tmp_path / "config.json").write_text("{}")
    return tmp_path


def test_weights_are_the_safetensors_files(tmp_path):
    assert weights_bytes(model(tmp_path, 5 * GB, 7 * GB)) == 12 * GB


def test_a_model_that_leaves_the_reserve_free_fits(tmp_path):
    m = model(tmp_path, 70 * GB)
    assert why_it_does_not_fit(m, device=lambda: (70 * GB + RESERVE_BYTES, 96 * GB)) is None


def test_a_model_that_eats_into_the_reserve_is_refused_with_the_three_numbers(tmp_path):
    m = model(tmp_path, 40 * GB, 41 * GB)
    why = why_it_does_not_fit(m, device=lambda: (int(77.8 * GB), 96 * GB))
    assert why.startswith("this model does not fit this Mac.")
    assert "needs about 81.0 GB for its weights" in why
    assert "can give a model 77.8 GB" in why and "of the 96.0 GB" in why
    assert "smaller model, or a lower-bit version" in why
