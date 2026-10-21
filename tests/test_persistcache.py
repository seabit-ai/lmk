from pathlib import Path

from lmk.persistcache import (MIN_FREE_DISK_BYTES, OtherModelsRecords, PersistentBlobStore, _file_name, _key_of,
                              cache_budget, model_identity, prepare_cache_root, trim_to_limit)


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
    prepare_cache_root(root, exclude_from_backup=excluded.append)
    assert (root / "CACHEDIR.TAG").read_text().startswith("Signature: 8a477f597d28d172789f06886806bc55")
    prepare_cache_root(root, exclude_from_backup=excluded.append)
    assert excluded == [root]


# design 2026-10-20-cache-limit-spans-models: cache.max_size is one limit for every model's records
def record(directory, name, size, used_at_s):
    import os
    directory.mkdir(parents=True, exist_ok=True)
    f = directory / f"record@{name}@kv_delta.safetensors"
    f.write_bytes(b"x" * size)
    os.utime(f, ns=(used_at_s * 10**9, used_at_s * 10**9))
    return f


def test_the_other_models_records_are_counted_whole_and_listed_oldest_first(tmp_path):
    root = tmp_path / "cache"
    record(root / "live", "mine", 300, 30)
    record(root / "old-a", "a1", 200, 10)
    record(root / "old-b", "b1", 100, 20)
    (root / "old-b" / "layout.json").write_text("{}" * 5)
    others = OtherModelsRecords(root, "live")
    assert others.bytes == 200 + 100 + 10
    assert others.oldest_used() == 10 * 10**9
    assert others.evict_oldest() == 200
    assert others.bytes == 110 and others.oldest_used() == 20 * 10**9


def test_evicting_a_models_last_record_removes_its_directory(tmp_path):
    root = tmp_path / "cache"
    record(root / "old-a", "a1", 200, 10)
    (root / "old-a" / "layout.json").write_text("{}")
    others = OtherModelsRecords(root, "live")
    others.evict_oldest()
    assert not (root / "old-a").exists() and others.bytes == 0 and others.oldest_used() is None


class Live:
    """The live model's side of trim_to_limit: records as (used_at, size), oldest first."""
    def __init__(self, records):
        self.records = sorted(records)

    def used(self):
        return sum(size for _, size in self.records)

    def oldest_used(self):
        return self.records[0][0] if self.records else None

    def evict_oldest(self):
        self.records.pop(0)


def test_the_oldest_record_anywhere_goes_first_whichever_model_it_belongs_to(tmp_path):
    root = tmp_path / "cache"
    record(root / "old-a", "a1", 100, 10)
    record(root / "old-a", "a2", 100, 40)
    others = OtherModelsRecords(root, "live")
    live = Live([(20 * 10**9, 100), (30 * 10**9, 100)])
    trim_to_limit(450, live.used, live.oldest_used, live.evict_oldest, others)   # 400 held: fits, nothing goes
    assert others.bytes == 200 and len(live.records) == 2
    trim_to_limit(300, live.used, live.oldest_used, live.evict_oldest, others)   # 400 held, 300 allowed: a1 (t=10) goes
    assert others.bytes == 100 and len(live.records) == 2
    trim_to_limit(200, live.used, live.oldest_used, live.evict_oldest, others)   # now live's t=20 goes, then t=30 ties to nothing older
    assert (others.bytes, [t for t, _ in live.records]) == (100, [30 * 10**9])
    assert live.used() + others.bytes <= 200


def test_a_tie_in_age_goes_to_the_other_model_and_the_live_model_can_empty_the_rest(tmp_path):
    root = tmp_path / "cache"
    record(root / "old-a", "a1", 100, 10)
    others = OtherModelsRecords(root, "live")
    live = Live([(10 * 10**9, 100)])
    trim_to_limit(100, live.used, live.oldest_used, live.evict_oldest, others)
    assert others.bytes == 0 and len(live.records) == 1      # the tie: the other model's goes
    trim_to_limit(0, live.used, live.oldest_used, live.evict_oldest, others)
    assert live.records == []                                  # nothing else to take: the live model's own go


def test_trimming_reports_what_it_took_from_other_models_once(tmp_path):
    root = tmp_path / "cache"
    record(root / "old-a", "a1", 100, 10)
    record(root / "old-b", "b1", 50, 20)
    others = OtherModelsRecords(root, "live")
    live = Live([])
    trim_to_limit(0, live.used, live.oldest_used, live.evict_oldest, others)
    assert others.take_evicted() == (2, 150, 2)    # records, bytes, directories emptied
    assert others.take_evicted() == (0, 0, 0)


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


def test_a_quantized_kv_cache_gets_its_own_directory_and_16_keeps_the_old_one(tmp_path):
    model = make_model(tmp_path)
    plain = model_identity(model, repo="org/m", revision="abc123")
    assert model_identity(model, repo="org/m", revision="abc123", kv_cache_bits=16) == plain  # nothing moves for existing users
    eight = model_identity(model, repo="org/m", revision="abc123", kv_cache_bits=8)
    assert eight != plain
    assert model_identity(model, repo="org/m", revision="abc123", kv_cache_bits=4) not in (plain, eight)
