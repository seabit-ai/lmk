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
