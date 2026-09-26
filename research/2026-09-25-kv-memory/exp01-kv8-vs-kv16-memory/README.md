# exp01：kv8 与 kv16 的内存并排——同一套读数，8k / 32k / 64k / 128k

日期 2026-09-25。m3u（M3 Ultra 60 核 GPU，96 GB）。lmk worktree 分支 `kv-memory`（main 8e0c13a），`ENGINE_COMMIT` 42a248c；
Python 环境与引擎用主 checkout 的 `.venv` / `.engine`（worktree 里没有；`run.sh` 的 `LMK_CHECKOUT`）。

## 要回答
spec-long-context exp02 量了 kv16 的内存（128k 解码时 MLX active + 缓冲池 52–54 GB，MLX peak 42–43 GB），exp01 没量 kv8。
"按内存自动选 KV 精度"要两边的实测。本实验在**同一套读数**下把两边并排量一遍（kv16 也重量，不拿旧数拼）：
1. 每档上下文解码时进程占多少：ps 的 resident（`lmk/memory.py` 的读法）、进程 phys_footprint、MLX active / 缓冲池（cache）/ peak；KV 字节。
2. 引擎为两种精度算出的窗口（context-fit）与 lmk 的"tokens in memory"上限（`token_budget`）。

## 条件
- `qwen3.8-27b-4bit`，草稿 dflash2（推荐组合），思考关，贪心，单请求；`kv_cache_bits` 8 与 16 两个条件，先 8 后 16，各一个临时 lmk（1236 端口，按 PID 杀）。
- 前缀：spec-long-context exp01/02 的 `prefix.py`（拷来，只改一处：`.venv`/`.engine` 从 `LMK_CHECKOUT` 取），同样 8,192 / 32,768 / 65,536 / 131,072 token。
- 每档每类（code、prose）：一次 warm（max_tokens 1；每档第一类冷算新增的前缀，第二类从 cache 还原），再 3 次 256 token 的解码（前缀已在 cache）。
  两个条件各用一个新的临时 cache 目录（cache 身份含 KV 位数）。
- **常驻服务**：owner 说"可以停 lmk"，控制者 22:49:03 `lmk down`，本实验全部在那之后跑，常驻停着，不做常驻轮询/作废（与 spec-long-context exp01/02 不同）。
  临时实例加载前查空闲内存（< 45 GB 不加载）。

## 方法（`run.sh`，自包含）
- **进程内读数**：`probe/sitecustomize.py` 经 PYTHONPATH 在临时 lmk 启动时被 Python 自动 import（不改 lmk/ 与引擎）：每 0.25 s 记一次
  `mx.get_active_memory / get_cache_memory / get_peak_memory`；包一层 `MlxEngine.generate`，请求开始时 `mx.reset_peak_memory()`，流结束时再记一行
  ——所以每个请求的 peak 是它自己的（cache 还原 + 尾巴 prefill + 解码）。
- **进程外读数**（`measure.py`）：每 0.5 s 读 `ps -o rss`（`lmk.memory.SystemMemory.resident_bytes`）与 `proc_pid_rusage` 的 phys_footprint，
  请求中取最大，请求结束 5 s 后取最后一个（"请求后"）。
  冒烟时发现：**ps 的 RSS 看不见 MLX 的 GPU 分配**（一个 2 GB 的 mx 数组：RSS 40 MB，footprint 2.33 GiB，与 `footprint` 工具一致）；
  常驻 lmk 的 RSS ≈ 17.6 GB 是 mmap 的权重。所以两个都记，内存账以 footprint 与 MLX 计数为准，RSS 一列是"lmk 现在读的那个数看得见多少"。
- KV 字节 = 引擎 context-fit 日志的 `full_kv` B/token × (prompt + completion) token。窗口与 `token_budget` 从临时 lmk 的启动日志与 status 读。
- `table.py` 出 `results.md`（中位数）。

## 对比基准
- spec-long-context exp02（kv16，旧读法 = status 的 `lmk_gpu_bytes` 即 active + cache，0.5 s 轮询取最大）：解码时 18.9 / 25.5 / 34.2 / 52.0 GB（关草稿）、
  20.9 / 27.7 / 36.7 / 54.2（开草稿）；MLX peak（进程启动以来，不是每请求）20.4/19.1、24.6/23.9、30.1/30.4、42.2/43.3。
- 引擎公式：KV kv8 34,816 / kv16 65,536 B/token；working_set 77.76 GiB、reserve 3 GiB、baseline 14.95 GiB；token_budget kv8 1,844,474 / kv16 979,877；两者窗口都是 262,144。
- 冒烟（8k，1 次）：kv8 解码时 active 18.0 + cache 1.2、peak 18.3 GB；kv16 footprint 20.5、active 18.4 + cache 1.5、peak 19.1 GB。

## 预期（跑之前写）
- **E1 引擎窗口与上限复现**：两者窗口都是 262,144；token_budget kv8 1,844,474、kv16 979,877（与旧日志逐位相同）。把握：高。
- **E2 kv16 重量与 exp02 同量级**：128k 解码 active + cache 52–54 GB、footprint 比它多约 0.5–1 GB；每请求 peak 42–43 GB。把握：中高（读法换了，peak 现在是每请求的）。
- **E3 kv8 在 128k 省的不只 KV 那 4 GB**：exp02 的 peak（42 GB）≈ 权重 17 + 约 3 份 KV（生成用的 cache、热 cache、落盘快照）；kv8 同理 ≈ 17 + 3 × 4.6 ≈ 31 GB。
  active + cache 与 footprint 在 128k 约 36–42 GB（kv16 52–54），即 128k 上 kv8 省 12–18 GB。32k：kv8 约 22–24、kv16 约 25–28。把握：中
  （"约 3 份 KV"是由 exp02 的一组数倒推的，没拆过）。
- **E4 缓冲池随上下文涨**：cache 在 8k 约 1–2 GB，128k 约 8–10 GB（kv8 约为 kv16 的一半到三分之二）。把握：中。
- **E5 ps RSS 基本不动**：两种精度各档都在 17.5–19 GB（权重 mmap + 进程本身），看不见 KV 与缓冲池。把握：高（冒烟已见）。
- **E6 解码速度与 spec-long-context 同**：kv8 开草稿 code 约 53 / 42 / 31 / 20 tok/s，kv16 约 54 / 50 / 42 / 31（只作健全检查，本实验不为速度设计）。把握：中高。

## 结果（2026-09-25 22:55–23:33，`raw/`，全表 `results.md`）
kv8 22:55–23:15（含 8k→128k 冷 prefill），kv16 23:15–23:33。96 个请求（每条件 8 个 warm + 24 个解码），解码全部 256 token；常驻全程停着。
前缀 token 数与 spec-long-context exp01/02 逐行相同（`raw/prefix.txt`）。加载后、第一个请求前 MLX active 两边都是 17.14 GB（权重 + 草稿器）。

解码请求（前缀在 cache，256 token；code + prose 各 3 次的中位数；GB = 10^9）：

| 上下文 | KV 字节 kv8 → kv16 | footprint 解码中 kv8 → kv16 | 其中 MLX 缓冲池（最高）| MLX peak（每请求）| footprint 请求后 5 s | 请求后 active / 缓冲池 | ps RSS（lmk 现在读的）|
|---|---|---|---|---|---|---|---|
| 8k | 0.3 → 0.6 | 20.7 → 22.1 | 1.8 → 2.6 | 18.3 → 19.1 | 19.7 → 20.6 | 17.7/1.3 → 17.9/1.6 | 18.4 → 19.0 |
| 32k | 1.2 → 2.2 | 24.4 → 28.4 | 4.5 → 7.5 | 20.9 → 23.9 | 21.6 → 23.6 | 18.6/2.2 → 19.6/3.3 | 19.4 → 20.3 |
| 64k | 2.3 → 4.3 | 29.1 → 36.2 | 7.5 → 13.3 | 24.3 → 30.4 | 23.9 → 28.1 | 19.8/3.3 → 21.7/5.5 | 20.6 → 22.6 |
| 128k | 4.6 → 8.6 | **38.9 → 55.1** | 14.9 → 27.2 | 31.2 → 43.3 | 28.5 → 36.7 | 22.0/5.6 → 26.0/9.8 | 23.1 → 27.0 |

冷 prefill（warm 请求里冷算新增前缀的那一个）：128k（65k→131k 冷算）footprint kv8 44.9 / kv16 52.5 GB，MLX peak 38.8 / 43.0。
引擎与 lmk：窗口两边都是 262,144；token_budget kv8 1,844,474、kv16 979,877（`LmkReady` 日志）。decode tok/s（code / prose）kv8 53.2/41.0、41.2/33.4、30.8/25.8、20.7/17.0；
kv16 56.5/44.0、46.2/36.7、43.2/30.9、32.8/23.7。

一个请求里的时间线（exp02 的 probe，kv16 128k，同形）：请求开始时 active 25.9（权重 17.1 + 上一请求留下的热 cache ≈ 1 份 KV）；从磁盘还原前缀的 1–2 s 里
active 冲到 43.3（再多约 2 份 KV），随后这些临时 buffer 进缓冲池（26 GB），引擎在 prefill 结束时 `clear_cache` 把它清到 0；解码中缓冲池 < 1 GB；
请求结束前落盘快照又把约 1.1 份 KV 放进缓冲池（9.4 GB），一直留到下一个请求。

对预期：
- **E1 命中**：窗口 262,144 / 262,144，token_budget 1,844,474 / 979,877，逐位相同。
- **E2 命中**：kv16 128k 解码中 active + cache 53.9 GB（旧 52–54），footprint 55.1；每请求 peak 43.3（旧 42–43）。
- **E3 命中，省得比押的上沿还多一点**：kv8 128k 解码中 footprint 38.9 GB（押 36–42），比 kv16 少 16.2 GB，其中 KV 本身只差 4.0 GB。
  "约 3 份 KV"得到佐证：两种精度的每请求 peak 斜率都是 3.0 × KV B/token（exp03 的拟合），footprint 斜率 4.1–4.2 × KV。32k：kv8 24.4（押 22–24）、kv16 28.4（押 25–28）。
- **E4 命中**：缓冲池 8k 1.8 / 2.6、128k 14.9 / 27.2 GB——比押的（8–10）大，kv8 约为 kv16 的 55%。
- **E5 命中**：ps RSS 18.4–27.0 GB，比 footprint 少 2–28 GB；128k 时漏掉的多达 28 GB（kv16）。lmk 现在的 `resident_bytes` 只在加载进度上用（看权重 mmap），
  拿它当"进程占多少内存"会严重低估。
- **E6 命中**：速度与 spec-long-context 同量级（kv16 32k code 46–48 对旧 49.6，64k 43.2 对 41.6——不同的冷算路径给出不同的贪心文本，见 SLC-010）。
- 没押到的：**引擎的 baseline（14.95 GiB = 16.05 GB）不含草稿器**——context-fit 在草稿器加载之前跑（`raw/serve-*.log` 第 4 行 fit、第 6–7 行加载草稿），
  加载后 active 17.14 GB，差 1.1 GB。
