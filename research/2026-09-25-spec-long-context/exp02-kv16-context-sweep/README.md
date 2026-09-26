# exp02：exp01 换成 16 位 KV（`kv_cache_bits: 16`）——长上下文下 decode 是不是更快、多占多少内存

日期 2026-09-25。m3u（M3 Ultra 60 核 GPU，96 GB）。lmk 分支 `spec-long-context-2`（main 090546d），`ENGINE_COMMIT` 42a248c。

## 要回答
exp01（SLC-004）关草稿 128k 一步 65 ms（kv8），而 exp13 的 bf16 微基准外推约 48 ms。owner 的假设：**kv8 的反量化代价随上下文变长而涨**。
同条件只把 KV 换成 16 位，量：
1. 关/开草稿的每步 ms、tok/s、加速比、接受率，与 exp01（kv8）并排；kv16 从哪一档起 decode 明显更快（或不快）。
2. 内存：每档的 KV 字节与进程占用；kv16 下 262k 窗口在本机装不装得下、剩多少余量——kv8 存在的理由就是省这块。

## 条件（与 exp01 相同，只改一个参数）
- `qwen3.8-27b-4bit`，**`kv_cache_bits: 16`**（exp01 是 8），草稿器 dflash2（4 位），思考关，贪心（`temperature: 0`），前缀已在 cache，
  每格 256 token × 3 次，8k / 32k / 64k / 128k × code / prose × 草稿关 / 开；32k 另加模型缺省采样（temp 1.0 / top_p 0.95 / top_k 20）各 3 次。
- 前缀与任务文本与 exp01 相同（`prefix.py` 原样拷贝；同一 `.venv` 里的同一批源码，切出来的 token 数应与 exp01 的 `raw/prefix.txt` 一致，跑完核对）。
- cache 身份含 KV 位数，kv8 的 cache 用不上：两个条件共用一个新的临时 cache 目录，关草稿那轮冷算前缀，开草稿那轮命中。

## 方法（`run.sh`，自包含）
- exp01 的脚本拷过来（exp01 的文件不动），差别：配置里 16 位；`measure.py` 多记两样——
  ① 临时 lmk 自己的 `draft.rounds` 请求前后之差（**精确轮数**；exp01 没有，是按 255 − 接受数估的）；
  ② 请求进行中每 0.5 s 读一次临时 lmk 的 `memory.lmk_gpu_bytes`，取最大值（进程的 GPU 占用：权重 + KV + 缓冲），请求后记 MLX 的 peak。
  多出来的这个轮询读的是 `mx.get_active_memory()` 一类的计数器，按理不影响 decode；exp01 没有它，比较时记着这一处差别。
- `table.py` 读 exp01 与 exp02 的 `runs.jsonl`，出并排表；KV 字节 = 引擎 context-fit 日志里的 `full_kv=…B/token` × token 数
  （kv8 34,816 B/token 来自 exp01 的 `serve-off.log`；kv16 的数从本次日志读，`raw/context-fit-*.txt`）。kv8 的进程占用 exp01 没量，
  表里那一列是**推算**（kv16 实测 − 两种 KV 字节之差），标明。
- 262k 装不装得下：读临时 lmk 启动时的 context-fit 日志（`fitted=`、working_set、baseline）与 `status` 的 `context_length`、`token_budget`。
  **不跑 262k 的请求**（冷 prefill 一次约 20 分钟以上，问题问的是装不装得下）；余量按引擎公式算，标明是公式。
- **常驻服务**：owner 21:24:09 说"可以停 lmk"，控制者 `lmk down` 停了常驻（不自己拉起）。本实验从那之后跑，常驻是停的；
  `measure.py` 仍每 0.5 s 探一次 1235（连不上记为 `resident_poll_errors`，这正是"常驻停着"的记录；若中途有人拉起且在答，照旧作废重跑）。
  跑之前查空闲内存（< 45 GB 不加载）。临时 lmk 在 1236，按 PID 杀。

## 对比基准
- exp01（同机同模型同题，kv8）：关草稿 35.3 / 27.6 / 21.6 / 15.4 tok/s（code 贪心，8k→128k），即每步 28 / 36 / 46 / 65 ms；
  开草稿 code 1.50 / 1.51 / 1.43 / 1.28×，prose 1.12 / 1.27 / 1.24 / 1.23×；32k 采样 prose 1.02×、code 1.40×。
- exp13 / SPD-019（bf16 KV，单次前向微基准，不经服务）：T=1 4k 29 ms、32k 33.5 ms，每 1k 上下文 +0.15 ms；外推 128k ≈ 48 ms。
  exp01 kv8 的斜率约 +0.30 ms / 1k（8k→128k：28 → 65 ms），是它的两倍。
- KVQ-004（短 prompt）：kv8 decode 只慢 1.5%（38.9 对 39.5 tok/s）；"长上下文下 decode 的差别未量"——本实验补这一格。
- 引擎的 context-fit（exp01 日志，kv8）：working_set 77.76 GiB、reserve 3 GiB、baseline 14.95 GiB、prompt_inputs 10,240 B/token、
  attention 48 B/context/step，fitted = 262,144。

## 预期（跑之前写）
推导：纯带宽账（权重 ~15 GB 每步读一遍，KV 每步读一遍；kv16 的 KV 字节约为 kv8 的 1.9 倍）说 **kv16 该更慢**：128k 上多读 4 GB，
按 ~800 GB/s 约 +5 ms。但 exp01 的 kv8 斜率是 exp13 bf16 斜率的 2 倍，说明 kv8 路径每 1k 上下文的代价不只是读字节——
mlx 的 bf16 单 token 注意力走融合的 sdpa vector kernel，量化 KV 走"quantized_matmul → softmax → quantized_matmul"几段拆开的算子，
每段都要按上下文长度读写中间结果。我押后者占上风。
- **P1 关草稿 kv16 在长上下文明显更快**：8k 与 kv8 差 ≤ 5%（约 28 ms），32k 约 33 ms、64k 约 38 ms、128k 约 48–52 ms（kv8：36 / 46 / 65）。
  即 128k 约 20 tok/s 对 15.4。"明显更快"从 32k 起（≥ 8%）。把握：中（带宽账指向相反方向；押的是 exp13 的斜率）。
- **P2 开草稿也跟着快，加速比与 kv8 同量级或略高**：128k code 1.3–1.45×、prose 1.2–1.3×。校验宽度 3–5 行查询，bf16 的 sdpa 在
  T ≤ 5 时仍走向量 kernel（SPD-015 的悬崖在 T ≥ 6）。把握：中低。
- **P3 接受率与 kv8 同一范围**（code 70–85%，prose 55–65%），不随上下文单调变。把握：中。
- **P4 贪心开/关文本 kv16 下多数格一致**（kv8 是 8 格全分叉，SLC-006 归因于量化注意力 L>1 / L=1 不位级一致；bf16 也有浮点平手，
  但少得多）：code 大多一致、prose 可能个别分叉。把握：中低——这一条同时检验 SLC-006 的归因。
- **P5 内存**：KV 16 位 65,536 B/token（16 层全注意力 × K/V × 4 头 × 256 维 × 2 字节），128k 约 8.6 GB（kv8 4.6 GB）；
  进程 128k decode 时约 17 + 8.6 ≈ 26 GB。**262k 装得下**：按引擎公式每 token 65,536 + 10,240 + 48 × 2048 ≈ 174 KB，
  262k 约 42.5 GiB，可用约 77.76 − 14.95 − 3 ≈ 59.8 GiB，余约 17 GiB；`token_budget` 从 1.84M 掉到约 0.9–1.0M。把握：高（公式）/ 中（实测进程占用）。
- 采样行（32k）：与 exp01 一样 prose 采样几乎不赚（≈ 1.0–1.1×）——kv 位数不碰采样那 10 ms。把握：中。
