"""A prefix cache that survives a model reload and a reboot (wish list WISH-001).

mlx-engine keeps its cache in an unnamed temporary file with the index in
memory, so every reload starts cold. This module swaps in two pieces with the
engine's own interfaces:

  PersistentBlobStore   — one file per record; "the file exists" IS "the record
                          exists". Write-to-temp + rename, so a crash leaves at
                          worst a stray temp file, never a half record.
  PersistentPromptCacheStore — the engine's store, plus: restore the index from
                          what is on disk at startup; keep the files on close.

The cache directory is keyed by model identity AND engine commit: KV tensors
from different weights (or a different storage layout) must never be restored.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

CACHE_FORMAT_VERSION = 1
_LAYOUT_FILE = "layout.json"
_SUFFIX = ".safetensors"


def model_identity(model_path: Path, engine_commit: str) -> str:
    """Changes whenever the weights, the model config or the engine change."""
    h = hashlib.sha256()
    h.update(f"lmk-cache-v{CACHE_FORMAT_VERSION}|{engine_commit}|".encode())
    for f in sorted(model_path.iterdir()):
        if f.suffix in (".safetensors", ".json") and f.is_file():
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}|".encode())
    return h.hexdigest()[:24]


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


def _layout_to_json(layout) -> dict:
    return {"layer_kinds": list(layout.layer_kinds),
            "layer_indices_by_kind": {k: list(v) for k, v in layout.layer_indices_by_kind.items()},
            "rotating_window_size": layout.rotating_window_size}


def make_persistent_store_class(cache_root: Path, model_path: Path, engine_commit: str):
    """Build the store class the engine will instantiate in place of its own.
    A class (not an instance) because the engine constructs it itself."""
    from mlx_engine.model_kit.batched_vision.prompt_cache.cache_store import VlmPromptCacheStore
    from mlx_engine.model_kit.batched_vision.prompt_cache.disk_budget import provisional_cache_store_budget_bytes
    from mlx_engine.model_kit.batched_vision.prompt_cache.types import PromptCacheLayout, PromptCacheRecordMetadata

    from lmk import log

    directory = cache_root / model_identity(model_path, engine_commit)

    class PersistentPromptCacheStore(VlmPromptCacheStore):
        def __init__(self, max_kv_size: Optional[int] = None, enable_disk_cache: bool = True):
            super().__init__(max_kv_size=max_kv_size, enable_disk_cache=False)  # skip the temp-file store
            self._base_dir = directory
            self._blob_store = PersistentBlobStore(directory)
            self._empirical_budget_set = False
            self._max_cache_store_bytes = provisional_cache_store_budget_bytes(directory)
            self._restore_index()

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
                tmp.write_text(json.dumps(_layout_to_json(self._layout)))
                os.replace(tmp, directory / _LAYOUT_FILE)

        def _touch_cache_entry(self, key: str) -> None:
            super()._touch_cache_entry(key)
            self._blob_store.touch(key)  # mtime carries the LRU order across restarts

        def close(self) -> None:
            pass  # keep index and files; the process is going away anyway

    return PersistentPromptCacheStore
