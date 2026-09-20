"""A prefix cache that survives a model reload and a reboot (wish list WISH-001).

mlx-engine keeps its cache in an unnamed temporary file with the index in
memory, so every reload starts cold. This module swaps in two pieces with the
engine's own interfaces:

  PersistentBlobStore   — one file per record; "the file exists" IS "the record
                          exists". Write-to-temp + rename, so a crash leaves at
                          worst a stray temp file, never a half record.
  PersistentPromptCacheStore — the engine's store, plus: restore the index from
                          what is on disk at startup; keep the files on close.

The cache directory is keyed by the model's weights and by OUR record format
version — not by the engine commit (design OOBE §E′): an engine upgrade that
leaves the records readable must not cost the user a 100 GB cache. Whether it
does is settled by an integration test at upgrade time; if it fails, bump
CACHE_FORMAT_VERSION.
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from lmk import log

CACHE_FORMAT_VERSION = 1
_LAYOUT_FILE = "layout.json"
_SUFFIX = ".safetensors"


def model_identity(model_path: Path, repo: Optional[str] = None, revision: Optional[str] = None) -> str:
    """Changes whenever the weights or the model config change. A HuggingFace
    snapshot is named by its commit hash; a local directory has no such name,
    so its files are fingerprinted."""
    h = hashlib.sha256()
    h.update(f"lmk-cache-v{CACHE_FORMAT_VERSION}|".encode())
    if revision:
        h.update(f"hf:{repo}@{revision}".encode())
    else:
        for f in sorted(model_path.iterdir()):
            if f.suffix in (".safetensors", ".json") and f.is_file():
                st = f.stat()
                h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}|".encode())
    return h.hexdigest()[:24]


_CACHEDIR_TAG = "Signature: 8a477f597d28d172789f06886806bc55\n# This directory holds lmk's prompt cache. It can be deleted; lmk rebuilds it.\n"


def _exclude_from_time_machine(directory: Path) -> None:
    try:
        subprocess.run(["tmutil", "addexclusion", str(directory)], check=False, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass  # not macOS, or tmutil unavailable: the tag file still marks it for other backup tools


def _dir_bytes_and_last_use(directory: Path) -> tuple[int, int]:
    total, newest = 0, 0
    for f in directory.rglob("*"):
        if f.is_file():
            st = f.stat()
            total += st.st_size
            newest = max(newest, st.st_mtime_ns)
    return total, newest


MIN_FREE_DISK_BYTES = 10 * 1024**3


def cache_budget(max_bytes: int, used_bytes: int, free_disk_bytes: int) -> int:
    """cache.max_size IS the limit (design OOBE §C2, revised). One guard on top of it:
    the disk keeps MIN_FREE_DISK_BYTES free — below that the cache stops growing and
    gives space back, least recently used first.

    Deliberately NOT the engine's own budget (one full context's worth, at most a quarter
    of the free disk): that sizes a temporary cache for one conversation. Ours is
    persistent and shared by weeks of conversations, and a limit the user wrote down
    must not be silently undercut by a formula."""
    return max(0, min(max_bytes, used_bytes + free_disk_bytes - MIN_FREE_DISK_BYTES))


def prepare_cache_root(cache_root: Path, live_identity: str, max_bytes: int,
                       exclude_from_backup: Callable[[Path], None] = _exclude_from_time_machine) -> int:
    """One size limit for the whole cache directory (design OOBE §C2). Directories
    of other identities — a model used before, a retired record format — are
    never touched again, so they are the least recently used: they go first,
    whole, until the directory fits. Returns what is left for the live model."""
    cache_root.mkdir(parents=True, exist_ok=True)
    tag = cache_root / "CACHEDIR.TAG"
    if not tag.exists():
        tag.write_text(_CACHEDIR_TAG)
        exclude_from_backup(cache_root)

    live_bytes = _dir_bytes_and_last_use(cache_root / live_identity)[0] if (cache_root / live_identity).is_dir() else 0
    others = []
    for d in cache_root.iterdir():
        if d.is_dir() and d.name != live_identity:
            size, last_use = _dir_bytes_and_last_use(d)
            others.append((last_use, size, d))
    others.sort()
    others_bytes = sum(size for _, size, _ in others)
    while others and live_bytes + others_bytes > max_bytes:
        _, size, d = others.pop(0)
        shutil.rmtree(d, ignore_errors=True)
        others_bytes -= size
        log.info("LmkCacheOrphanDropped", "dropped the cache of a model no longer in use to fit cache.max_size",
                 dir=str(d), mib=round(size / 1048576, 1))
    return max(0, max_bytes - others_bytes)


# Keys look like "record:<sha256>:<kind>", and kinds contain underscores
# (state_checkpoint), so ":" maps to a character that appears in neither.
_KEY_SEP_IN_FILE = "@"


def _file_name(key: str) -> str:
    return key.replace(":", _KEY_SEP_IN_FILE) + _SUFFIX


def _key_of(file_name: str) -> str:
    return file_name[: -len(_SUFFIX)].replace(_KEY_SEP_IN_FILE, ":")


class PersistentBlobStore:
    """Same surface as the engine's TemporarySafetensorBlobStore. Not
    thread-safe, like the original: access is owned by the cache I/O thread."""

    def __init__(self, directory: Path):
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        for stray in self._dir.glob("*.tmp"):
            stray.unlink(missing_ok=True)

    def keys_oldest_first(self) -> list[str]:
        files = [f for f in self._dir.iterdir() if f.name.endswith(_SUFFIX)]
        files.sort(key=lambda f: f.stat().st_mtime_ns)
        return [_key_of(f.name) for f in files]

    def put(self, key: str, arrays: dict[str, Any], safetensor_metadata: dict[str, str]) -> int:
        import mlx.core as mx

        path = self._dir / _file_name(key)
        if path.exists():
            return path.stat().st_size
        tmp = path.with_suffix(".tmp")
        # mx.save_safetensors appends ".safetensors" to a bare path; hand it a file object
        with open(tmp, "wb") as f:
            mx.save_safetensors(f, arrays, safetensor_metadata)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path.stat().st_size

    def load_record(self, key: str) -> list[Any]:
        from mlx_engine.model_kit.batched_vision.prompt_cache.blob_store import _load_record_from_file

        with open(self._dir / _file_name(key), "rb") as f:
            return _load_record_from_file(f)

    def exists(self, key: str) -> bool:
        return (self._dir / _file_name(key)).exists()

    def size(self, key: str) -> int:
        return (self._dir / _file_name(key)).stat().st_size

    def touch(self, key: str) -> None:
        try:
            os.utime(self._dir / _file_name(key))
        except FileNotFoundError:
            pass

    def delete(self, key: str) -> None:
        (self._dir / _file_name(key)).unlink(missing_ok=True)

    def close(self) -> None:
        pass  # the whole point: the records stay


def _layout_to_json(layout, engine_commit: str) -> dict:
    return {"written_by_engine_commit": engine_commit,  # for the record only; not part of the identity
            "layer_kinds": list(layout.layer_kinds),
            "layer_indices_by_kind": {k: list(v) for k, v in layout.layer_indices_by_kind.items()},
            "rotating_window_size": layout.rotating_window_size}


def make_persistent_store_class(directory: Path, max_bytes: int, engine_commit: str, created: list):
    """Build the store class the engine will instantiate in place of its own.
    A class (not an instance) because the engine constructs it itself; the
    instance is appended to `created` so lmk can report its size."""
    from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import VlmPromptCacheStore
    from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptCacheLayout, PromptCacheRecordMetadata

    class PersistentPromptCacheStore(VlmPromptCacheStore):
        def __init__(self, max_kv_size: Optional[int] = None, enable_disk_cache: bool = True):
            super().__init__(max_kv_size=max_kv_size, enable_disk_cache=False)  # skip the temp-file store
            self._base_dir = directory
            self._blob_store = PersistentBlobStore(directory)
            self._empirical_budget_set = False
            self._max_cache_store_bytes = max_bytes  # _restore_index evicts against the real budget below
            self._restore_index()
            created.append(self)

        def _free_disk_bytes(self) -> int:
            return shutil.disk_usage(directory).free

        def commit_budget_update(self, max_cache_store_bytes: int) -> None:
            # the engine's estimate is ignored on purpose: see cache_budget
            super().commit_budget_update(cache_budget(max_bytes, self._total_bytes, self._free_disk_bytes()))

        def _evict_if_needed(self) -> None:
            self._max_cache_store_bytes = cache_budget(max_bytes, self._total_bytes, self._free_disk_bytes())
            super()._evict_if_needed()

        def stats(self) -> dict:
            return {"dir": str(directory), "used_bytes": self._total_bytes, "max_bytes": max_bytes,
                    "records": len(self._record_metadata_by_key),
                    "disk_low": self._free_disk_bytes() < MIN_FREE_DISK_BYTES}

        def _restore_index(self) -> None:
            layout_path = directory / _LAYOUT_FILE
            keys = self._blob_store.keys_oldest_first()
            if not layout_path.exists():
                # records without a layout cannot be interpreted: start clean
                for key in keys:
                    self._blob_store.delete(key)
                log.info("LmkCacheRestored", "no prior cache for this model", dir=str(directory), records=0)
                return
            raw = json.loads(layout_path.read_text())
            self._layout = PromptCacheLayout(layer_kinds=raw["layer_kinds"],
                                             layer_indices_by_kind=raw["layer_indices_by_kind"],
                                             rotating_window_size=raw["rotating_window_size"])
            for key in keys:  # oldest first → LRU order is rebuilt as we touch
                _, chunk_key, record_kind = key.split(":", 2)
                self._record_metadata_by_key[key] = PromptCacheRecordMetadata(
                    chunk_key=chunk_key, record_kind=record_kind,
                    layer_indices=list(self._layout.layer_indices_by_kind.get(record_kind, [])))
                super()._touch_cache_entry(key)
            log.info("LmkCacheRestored", "prefix cache restored from disk", dir=str(directory),
                     records=len(keys), mib=round(self._total_bytes / 1048576, 1))
            self._evict_if_needed()

        def commit_pending_save(self, pending_save) -> None:
            had_layout = self._layout is not None
            super().commit_pending_save(pending_save)
            if not had_layout and self._layout is not None:
                tmp = directory / (_LAYOUT_FILE + ".tmp")
                tmp.write_text(json.dumps(_layout_to_json(self._layout, engine_commit)))
                os.replace(tmp, directory / _LAYOUT_FILE)

        def _touch_cache_entry(self, key: str) -> None:
            super()._touch_cache_entry(key)
            self._blob_store.touch(key)  # mtime carries the LRU order across restarts

        def close(self) -> None:
            pass  # keep index and files; the process is going away anyway

    return PersistentPromptCacheStore
