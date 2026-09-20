from pathlib import Path

from lmk.persistcache import PersistentBlobStore, _file_name, _key_of, model_identity


def make_model(tmp_path, weights=b"w1", config=b"{}"):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "model-00001.safetensors").write_bytes(weights)
    (d / "config.json").write_bytes(config)
    (d / "README.md").write_text("ignored")
    return d


def test_identity_is_stable_and_changes_with_weights_config_and_engine(tmp_path):
    model = make_model(tmp_path)
    base = model_identity(model, "engine-a")
    assert model_identity(model, "engine-a") == base
    assert model_identity(model, "engine-b") != base
    (model / "README.md").write_text("still ignored")
    assert model_identity(model, "engine-a") == base
    (model / "model-00001.safetensors").write_bytes(b"different weights")
    assert model_identity(model, "engine-a") != base


def test_record_keys_round_trip_through_file_names():
    key = "record:" + "a" * 64 + ":state_checkpoint"
    assert ":" not in _file_name(key)
    assert _key_of(_file_name(key)) == key


def test_store_lists_what_is_on_disk_oldest_first_and_survives_reopen(tmp_path):
    import os
    store = PersistentBlobStore(tmp_path / "c")
    for i, key in enumerate(["record:k1:kv_delta", "record:k2:kv_delta"]):
        p = tmp_path / "c" / _file_name(key)
        p.write_bytes(b"x" * (i + 1))
        os.utime(p, ns=(10**9 * (i + 1), 10**9 * (i + 1)))
    assert store.exists("record:k1:kv_delta") and store.size("record:k2:kv_delta") == 2
    store.close()

    reopened = PersistentBlobStore(tmp_path / "c")
    assert reopened.keys_oldest_first() == ["record:k1:kv_delta", "record:k2:kv_delta"]
    reopened.touch("record:k1:kv_delta")  # most recently used moves to the end
    assert reopened.keys_oldest_first() == ["record:k2:kv_delta", "record:k1:kv_delta"]
    reopened.delete("record:k2:kv_delta")
    assert reopened.keys_oldest_first() == ["record:k1:kv_delta"]


def test_a_crash_mid_write_leaves_no_half_record(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d / "record@k9@kv_delta.tmp").write_bytes(b"partial")
    store = PersistentBlobStore(d)
    assert store.keys_oldest_first() == []
    assert not list(d.glob("*.tmp"))
