from pathlib import Path

from lmk.persistcache import (MIN_FREE_DISK_BYTES, PersistentBlobStore, _file_name, _key_of, cache_budget,
                              model_identity, prepare_cache_root)


def make_model(tmp_path, weights=b"w1", config=b"{}"):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "model-00001.safetensors").write_bytes(weights)
    (d / "config.json").write_bytes(config)
    (d / "README.md").write_text("ignored")
    return d


def test_a_local_model_is_identified_by_its_weight_and_config_files(tmp_path):
    model = make_model(tmp_path)
    base = model_identity(model)
    assert model_identity(model) == base
    (model / "README.md").write_text("still ignored")
    assert model_identity(model) == base
    (model / "model-00001.safetensors").write_bytes(b"different weights")
    assert model_identity(model) != base


def test_a_huggingface_model_is_identified_by_repo_and_commit_not_by_its_files(tmp_path):
    model = make_model(tmp_path)
    base = model_identity(model, repo="org/m", revision="abc123")
    (model / "model-00001.safetensors").write_bytes(b"touched, same commit")
    assert model_identity(model, repo="org/m", revision="abc123") == base
    assert model_identity(model, repo="org/m", revision="def456") != base
    assert model_identity(model, repo="other/m", revision="abc123") != base


def fill(directory, size, used_at_s):
    import os
    directory.mkdir(parents=True)
    f = directory / "record@k@kv_delta.safetensors"
    f.write_bytes(b"x" * size)
    os.utime(f, ns=(used_at_s * 10**9, used_at_s * 10**9))


def test_cache_root_is_tagged_and_excluded_from_backup_once(tmp_path):
    excluded = []
    root = tmp_path / "cache"
    assert prepare_cache_root(root, "live", 1000, exclude_from_backup=excluded.append) == 1000
    assert (root / "CACHEDIR.TAG").read_text().startswith("Signature: 8a477f597d28d172789f06886806bc55")
    prepare_cache_root(root, "live", 1000, exclude_from_backup=excluded.append)
    assert excluded == [root]


def test_other_models_caches_are_kept_while_everything_fits(tmp_path):
    root = tmp_path / "cache"
    fill(root / "live", 300, used_at_s=30)
    fill(root / "old-a", 200, used_at_s=10)
    fill(root / "old-b", 100, used_at_s=20)
    assert prepare_cache_root(root, "live", 1000, exclude_from_backup=lambda d: None) == 700
    assert (root / "old-a").is_dir() and (root / "old-b").is_dir()


def test_over_the_limit_other_models_caches_go_first_longest_unused_first(tmp_path):
    root = tmp_path / "cache"
    fill(root / "live", 300, used_at_s=30)
    fill(root / "old-a", 200, used_at_s=10)
    fill(root / "old-b", 100, used_at_s=20)
    assert prepare_cache_root(root, "live", 450, exclude_from_backup=lambda d: None) == 350
    assert not (root / "old-a").exists() and (root / "old-b").is_dir() and (root / "live").is_dir()


def test_the_live_cache_is_never_dropped_here_even_when_alone_over_the_limit(tmp_path):
    root = tmp_path / "cache"
    fill(root / "live", 300, used_at_s=30)
    fill(root / "old-a", 200, used_at_s=10)
    assert prepare_cache_root(root, "live", 100, exclude_from_backup=lambda d: None) == 100
    assert not (root / "old-a").exists() and (root / "live").is_dir()  # the store trims itself, record by record


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


GB = 1024**3


def test_the_configured_size_is_the_limit_whatever_the_disk_could_hold():
    assert cache_budget(200 * GB, used_bytes=10 * GB, free_disk_bytes=300 * GB) == 200 * GB
    assert cache_budget(200 * GB, used_bytes=0, free_disk_bytes=4000 * GB) == 200 * GB


def test_the_disk_keeps_its_last_ten_gigabytes():
    assert MIN_FREE_DISK_BYTES == 10 * GB
    # 50 used, 30 free: the cache may grow by 20 more, then the disk is at its floor
    assert cache_budget(200 * GB, used_bytes=50 * GB, free_disk_bytes=30 * GB) == 70 * GB
    # at the floor exactly: no growth
    assert cache_budget(200 * GB, used_bytes=50 * GB, free_disk_bytes=10 * GB) == 50 * GB


def test_below_the_floor_the_cache_gives_space_back():
    # 4 GB free: the cache must shrink by 6 to restore the floor
    assert cache_budget(200 * GB, used_bytes=50 * GB, free_disk_bytes=4 * GB) == 44 * GB
    assert cache_budget(200 * GB, used_bytes=2 * GB, free_disk_bytes=1 * GB) == 0
