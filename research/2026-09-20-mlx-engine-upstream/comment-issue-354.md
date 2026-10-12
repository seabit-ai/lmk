+1 with data — we run this in production, built outside the engine behind the existing store interface.

Qwen3.8-27B-MLX-4bit on an M3 Ultra, coding-agent workload, ~11.2k-token stable prefix:

| first request of a new session | tokens restored | time to first token |
|---|---|---|
| cold cache | 0 / 11,175 | 36.8 s |
| **after a full process restart** | **11,008 / 11,174** | **2.4 s** |

Three weeks in production since: 98% of 84.5M prompt tokens served from cache; an 82k-token conversation reconnects in ~3 s.

Today every restart pays the cold row, because the blob store is an unnamed tempfile and its index lives in memory. Persistence needed no change to the chunking or the restore planner (chunk keys are already content hashes): a file-per-record blob store with the same surface, plus rebuilding the index from the directory at load. ~260 lines, MIT: <https://github.com/seabit-ai/lmk/blob/main/lmk/persistcache.py>. The only ugly part is installing it — we monkeypatch the store class in `model_kit.py`, since the kit constructs its store itself.

Re "needs to be wired into the actual frontend": it doesn't — it fits behind `load_model` arguments, and a frontend that passes nothing gets today's behaviour.

Would you take a PR? Two steps:
1. `load_model(..., prompt_cache_dir: Path | None = None)` threaded to `VlmPromptCacheStore`, replacing the hard-coded `/tmp` (default unchanged);
2. an opt-in `persist_prompt_cache: bool = False` selecting the persistent store.
