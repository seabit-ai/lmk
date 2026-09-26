# exp03：采样的一轮投机为什么比贪心贵约 10 ms——拆开计时（只量，不修）

日期 2026-09-25。m3u（M3 Ultra 60 核 GPU，96 GB）。lmk 分支 `spec-long-context-2`（main 090546d），`ENGINE_COMMIT` 42a248c，mlx-vlm 0.6.16。

## 要回答
exp01（SLC-005，32k）：采样散文一轮约 74 ms、贪心 64 ms（每轮起草都是 2.0，轮数是按 255 − 接受数**估的**）；采样 prose 加速 1.02×，贪心 1.27×。
1. 一轮拆成：起草、校验前向、逐位置采样（logsumexp / top-k / top-p 的排序 / categorical）、主机同步（`.tolist()`、`mx.eval`）、回滚、walk 的记账——多出来的 ms 在哪。
2. **只作估算**：若把一块的所有位置一次性在 GPU 上采完（向量化，一轮一次同步），能省多少、散文采样的加速比会到多少。哪些是量的、哪些是估的，分开写。

## 读代码得到的结构（42a248c / mlx-vlm 0.6.16）
- `SpeculativeGenerationBatch._round` → `dflash_round`：`draft_block`（DFlash2 一次前向 + CandidateSelector）→ `_verify_block`（目标模型一次批量前向，带
  `GdnVerifyRecorder`）→ walk → `_record_speculative_round` / `truncate` → 不全接受时 `rollback_speculative_cache`。
- 贪心 walk（`common._speculative_walk`）：`draft.tolist()`（同步 1：只算起草）、`argmax(logits).tolist()`（同步 2：校验前向 + argmax）。
- 采样 walk（`dflash._sample_dflash_target_walk`；引擎的 `create_sampler` 没有 `sample_target`，走**非定位**分支）：`draft.tolist()`（同步 1），
  `_dflash_target_logprobs` 对每个位置各做一次 logsumexp 再 stack，然后**逐位置**：`sampler(logprobs[:, pos])`（top_p：全词表 argsort + cumsum +
  scatter；top_k：argpartition；categorical）→ `mx.eval` → `.tolist()` → 与草稿比，不同就停。每走一个位置一次同步；第一个位置的 eval 里包着校验前向。
- 起草两边一样：CandidateSelector 只认 `sample_proposal`，引擎的采样器没有，两边都是 argmax。

## 方法（`run.sh`，自包含）
- 一个临时 lmk（1236，按 PID 杀；`profile_serve.py` = 先装 `profile_patch.py` 再跑 `lmk serve`，同一进程）。补丁把引擎模块里的
  `dflash_round` 换成它的**拷贝加计时**（mlx-vlm 的采样 walk 内联进来），并包住 `SpeculativeGenerationBatch.next` 计整步与步间间隔。
  lmk/ 与引擎的文件不动。
- 条件 = exp01 的"开草稿"：`qwen3.8-27b-4bit`、kv8、dflash2、思考关，exp01 的 32k 前缀（`prefix.py` 原件）与两个任务，256 token；
  贪心（`temperature: 0`）vs 模型缺省采样（temp 1.0 / top_p 0.95 / top_k 20，请求不带 temperature）；每格 3 次。
- 两种计时模式（每个请求一种，控制文件切换）：
  - **wall**：拷贝里的操作与顺序和引擎一模一样，计时只读时钟；同步点就是引擎自己的那几个。某个同步"等了多久" = 在它之前排进 GPU 的活。
    这是"多出来的 ms 在哪"的主证据。
  - **phases**：每段后面加一次 `mx.eval`，得到每段自己的 GPU 时间（+ 一次同步）；加的同步改了调度，段之和 ≠ wall 的一轮。
    这个模式下每个请求的前 6 个"校验宽 ≥ 3"的轮，在该轮计时结束后，拿**这一轮真实的校验 logits** 做微基准：引擎式逐位置采完全部位置
    vs 一次 logsumexp + 采样器吃 [位置, 词表] + 一次 eval；以及单个位置的 logsumexp / top_p / top_k / categorical / 空同步各多少（各 5 次取中位数）。
- 常驻服务：owner 21:24:09 说"可以停 lmk"，控制者已 `lmk down`；本实验全程常驻停着。`measure.py` 仍照 exp01 探 1235（连不上 = 停着的记录）。
  跑前查空闲内存（< 40 GB 不加载）。exp02 跑完才开始，两个临时实例不同时在。
- `analyze.py` 出三张表：请求级（tok/s、精确轮数、一轮 ms）；wall 拆分；phases 拆分；微基准。

## 对比基准
- exp01 32k（kv8，同条件不带补丁）：开草稿贪心 prose 35.3 tok/s、采样 prose 28.1；code 41.6 / 38.3。一轮（估）prose 64 / 74 ms，code 94 / 73 ms
  （code 采样每轮起草 2.4、贪心 3.5，块不同宽，不可直接比）。**关草稿**采样与贪心每步只差约 0.3 ms（36.4 对 36.1 ms，prose）——
  普通 decode 每步也要按同一个采样器采一次，所以**一次采样的 GPU 代价上限约 0.3–0.5 ms**。
- exp13 / SPD-019（bf16，32k）：校验前向 T=1 33.5 ms、T=5 69.8 ms；引擎一轮 DFlash2 约 71 ms（草稿 15 + 校验 55 + 回滚 1）。

## 预期（跑之前写）
- **P1 wall 模式下，采样 prose 的一轮比贪心 prose 贵 6–12 ms**（exp01 估 10）。把握：中（exp01 的轮数是估的）。
- **P2 采样算子本身不是大头**：每走一个位置的采样（logsumexp + top_p 排序 + top_k + categorical）GPU 约 0.3–1 ms，一轮走约 2 个位置，
  合计 ≤ 2 ms。依据是关草稿时采样与贪心每步只差 0.3 ms。把握：中。
- **P3 大头在同步与调度**：采样每多走一个位置就多一次"GPU 做完 → 主机取回 → 主机再排下一段"，外加 top_p 的全词表 argsort 在
  [1, 248k] 上是很窄的并行度。我押 wall 里"第 2 个位置起"的时间 + 第一个位置 eval 相对贪心 argmax 的多出部分合计能解释多出 ms 的一半以上；
  剩下的若在 round 函数之外（步间间隔、`_round` 的其余部分），会在 step / gap 两列里看到。把握：低——这正是要量的。
- **P4 向量化一次采完（估算）**：一轮省下"额外位置的同步 + 逐位置 logsumexp"，采样的一轮落到贪心一轮 + 1–2 ms；
  按 exp01 采样 prose 每轮吐 2.08 个 token、关草稿 36.4 ms/token 推，散文采样加速从 1.02× 到约 1.1–1.15×（接受率 54% 不变：
  向量化不改"逐位置采目标 token、与草稿比"的规则，分布相同）。把握：低（估算套估算）。要更多，得换接受规则（按 p/q 的拒绝采样），不在本实验。
- P5 phases 模式的段之和比 wall 一轮大（多了同步），差值约等于多加的同步次数 × 空同步耗时。把握：中。
