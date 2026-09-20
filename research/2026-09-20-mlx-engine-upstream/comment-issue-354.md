+1, with data — we built this outside the engine and it works, and the change needed inside the engine looks small.

**Setup.** A small single-model server on top of mlx-engine (pinned at `08f0c07`, mlx 0.32.0), serving Qwen3.8-27B-MLX-4bit on an M3 Ultra to a coding agent. The agent's fixed prefix (tool definitions + system prompt) is ~11.2k tokens, byte-stable across sessions.

**Result.** With the prompt cache made persistent:

| first request of a new session | tokens restored | time to first token |
|---|---|---|
| cold cache | 0 / 11,175 | 36.8 s |
| **after a full process restart** | **11,008 / 11,174** | **2.4 s** |

Today every model reload or restart pays the cold row again, because the store is an unnamed `tempfile.TemporaryFile` and the index lives only in memory.

**What we had to do.** `VlmPromptCacheStore.__init__` hard-codes `base_dir = Path("/tmp")` and constructs `TemporarySafetensorBlobStore` itself, and `BatchedVisionModelKit` constructs the store itself — so the only way in was to replace the class name in `model_kit/batched_vision/model_kit.py` before `load_model`. That works, but it is obviously fragile.

**What persistence took** (all of it fits behind the existing store interface):
- A blob store with the same surface (`put / load_record / exists / size / delete / close`) that keeps **one file per record**, written to a temp name and renamed — "the file exists" is "the record exists", so there is no extent index to persist and a crash leaves at worst a stray temp file.
- At startup, rebuild `_record_metadata_by_key`, sizes and LRU order from the directory. The record key already encodes the chunk hash and record kind; layer indices come from the `PromptCacheLayout`, which we persist once as a small JSON. LRU order survives via file mtime (touch on access).
- `close()` keeps the files.
- The directory is keyed by **model identity** (names/sizes/mtimes of the weight and config files) **plus the engine commit**, so KV from different weights or a different storage layout is never restored.

Nothing about the chunking or the restore planner had to change — chunk keys are already content hashes, which is exactly what makes this safe.

On the concern in the issue that this "needs to be wired into the actual frontend": it doesn't have to. It can live entirely in the engine behind `load_model` arguments; a frontend that passes nothing gets today's behaviour.

**The ask.** Would you accept a PR along these lines? Happy to split it:
1. *Injection point only:* `load_model(..., prompt_cache_dir: Path | None = None)` threaded to `VlmPromptCacheStore`, replacing the hard-coded `/tmp`. Default unchanged. (This alone would also cover #335.)
2. *Persistent store:* an opt-in `persist_prompt_cache: bool = False` that selects the file-per-record store and restores the index on load.

Two things we ran into that a real implementation should decide: the disk budget is currently derived from free space / `max_kv_size` (we see `cap_gib=162.81`), which is fine for a temp file but probably wants an explicit cap once the data outlives the process; and the identity key should include a storage-format version so a layout change invalidates old directories cleanly.

Glad to share the code or adapt it to whatever shape you prefer.
