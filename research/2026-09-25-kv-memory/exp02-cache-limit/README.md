# exp02：给 MLX 的缓冲池设上限（或请求结束时清掉），能不能把那约 10 GB 还回去而不拖慢 decode

日期 2026-09-25。m3u（M3 Ultra 60 核 GPU，96 GB）。lmk worktree 分支 `kv-memory`（main 8e0c13a），`ENGINE_COMMIT` 42a248c，mlx 0.32.0；
Python 环境与引擎用主 checkout 的 `.venv` / `.engine`。

## 要回答
spec-long-context exp02：kv16 在 128k 解码时 MLX active + 缓冲池 52–54 GB，peak in use 42–43 GB——约 10 GB 是 MLX 的缓冲池（释放了但留着复用的
Metal buffer）。lmk 与引擎都没调 `mx.set_cache_limit`（缺省 = memory limit，即不设限）；引擎只在 prefill 结束、每 512 步、卸载时 `mx.clear_cache()`。
1. `mx.set_cache_limit(N)`（N = 4 GB、1 GB、0）让进程在请求中与请求后各占多少（footprint、active、cache、peak）？
2. 请求结束时 `mx.clear_cache()` 一次（不设上限）效果如何？
3. decode tok/s（每格 3 次）与 TTFT（从磁盘 cache 还原前缀）变不变？

## 条件
- `qwen3.8-27b-4bit`，`kv_cache_bits: 16`（本机现行），草稿 dflash2，思考关，贪心，单请求，code 任务；32k 与 128k；每档 1 次 warm（max_tokens 1）+ 3 次 256 token 解码。
- 六个条件，各一个临时 lmk（1236，按 PID 杀），依次：`unset1`（不设限）→ `limit0` → `limit1g` → `limit4g` → `clear`（不设限，请求结束清一次）→ `unset2`（基线重来一遍，看漂移）。
  共用一个临时 cache 目录：只有 `unset1` 的 warm 冷算前缀，其余的 warm 从磁盘还原。
- 上限在临时 lmk 进程里、引擎加载之前设：`probe/sitecustomize.py`（exp01 那份，读 `LMK_EXP_CACHE_LIMIT` / `LMK_EXP_CLEAR_AT_END`），不改 lmk/ 与引擎。
  它同时记下缺省上限（`set_cache_limit` 的返回值）。
- **常驻服务停着**：owner 说"可以停 lmk"，控制者 22:49:03 `lmk down`；本实验在那之后跑，不做常驻轮询。

## 方法
与 exp01 相同的 `measure.py`（拷来）与读数：进程外 0.5 s 一次 ps RSS 与 phys_footprint，进程内 0.25 s 一次 MLX active / cache / peak，
peak 每请求重置；"请求后" = 响应后 5 s 的最后一次读数。`table.py` 出 `results.md`，decode 速度逐次列出。

## 对比基准
- spec-long-context exp02（kv16，开草稿，code 贪心）：128k decode 31.2 tok/s、32k 49.6（每格 3 次，开草稿贪心离散 ≤ 12%）；
  active + cache 128k 54.2、32k 27.7 GB；peak 43.3 / 23.9 GB。
- exp01（本组，同一 harness，kv16 行）——跑完后对照。

## 预期（跑之前写）
- **C1 上限把缓冲池压到上限以下，请求中的 footprint 降到约 active + 上限 + 0.8**：128k 不设限约 53–55 GB；4 GB 约 46–48；1 GB 约 43–45；0 约 42–44（≈ active）。
  32k 同理但差得少（缓冲池约 3–4 GB）。把握：中高（MLX 文档写"超过上限时在下一次分配回收"）。
- **C2 请求后**：不设限时缓冲池留在进程里（footprint 回不到 active）；设了上限的都 ≤ active + 上限；`clear` 在请求后 ≈ active（与 `limit0` 同），但请求中与不设限相同。
  请求后的 active 里还有引擎的热 cache（上一请求的 KV），不归缓冲池管。把握：中高。
- **C3 decode 速度**：4 GB / 1 GB 与不设限差 ≤ 3%（解码每步的临时 buffer 在 128k 也只有几十 MB，1 GB 装得下，复用照旧）。`limit0` 每次分配都向 Metal 要新 buffer，
  每步几百次分配 × 约 10–50 µs → 128k 每轮（约 128 ms）多几 ms，慢 3–10%；32k 相对影响更大（一轮更短）。把握：1 GB / 4 GB 中高，`limit0` 低（分配代价没量过）。
- **C4 TTFT（从磁盘还原前缀）**：设限对还原慢不了多少（≤ 10%），`limit0` 可能慢一些。把握：低。
- **C5 peak in use 不变**（上限只管已释放的 buffer，不管同时在用的）：128k 约 42–44 GB，32k 约 24 GB。把握：高。
- 结论若 C1–C3 成立：lmk 可以在加载前设一个 1–4 GB 的缓冲池上限，128k 时把进程占用从约 53 GB 降到 43–47 GB，代价可忽略。
