# seed 可重放：取证与实现记录（SEED）

设计与裁决在 `docs/design/2026-09-26-sampling-seed.md`；这里放依据、实现细节、没量过的东西，和留给 owner 跑的集成测试。

起因：kitten 拿到一个坏答案（qwen3.8-27b-4bit，temp 1.0 / top_p 0.95 / top_k 20，dflash2，kv16，max_parallel 2），无法复现。
**那一次本身重放不了**：它发生在这个改动之前，没有 seed。这个改动让以后的每一次都能重放。

## 分支
- 引擎：fork `seabit-ai/mlx-engine` 分支 `lmk-seed`，基于 `42a248c`：`bab3536`、`c2c6f9e`（未 push）。
- lmk：分支 `sampling-seed`，`ENGINE_COMMIT` 指向 fork `lmk-seed` 的最新 commit（第一轮 bab3536，第二轮 c2c6f9e；未 push，未合）。
  **合并前 owner 先 push fork**，否则 install.sh 按 hash 取 tarball 会失败（CLAUDE.md 地图 `.engine/mlx-engine` 一行）。
- 工作树在 `~/src/lmk-seed`（引擎是其下 `.engine/mlx-engine`，另一个 git worktree，自带 `.venv`），没碰 `~/src/lmk` 与它的引擎工作树。

## 发现
- **SEED-001** 上游批处理路径（`generate._batched_generation`）收 `seed` 参数但不用（"Seed arg is ignored for batched gen"）；
  采样器是每请求一个（`create_sampler`），但调用 `mx.random.categorical` 走全局 PRNG，所以一行的随机数随批里别的行、
  随调用次数挪动。
- **SEED-002** mlx-vlm 0.6.16 已有"按位置抽样"的协议：采样器若有 `sample_target(logprobs, row_ids=, positions=)`，
  它的 MTP / DFlash 校验 walk 就按位置调用（`mtp._speculative_walk_batch_deferred_greedy`、`dflash._sample_dflash_target_walk`），
  并有 `_PositionedDraftSampler` 给草稿提议用。fork 的 `_processed_walk`（带 processor 的逐位 walk）已经按这个协议传位置；
  但 fork 调 `_speculative_walk_batch_deferred_greedy` 时**没传 `row_ids/base_positions`**——换上任何带 `sample_target`
  的采样器都会在那里抛 `ValueError`。已补。
- **SEED-003** 位置的约定（生成序列里的下标，从 0 起）在 fork 里本来就一致：投机轮的 `emitted = row.num_tokens`（已发出的个数，
  含 bonus）；walk 第 j 位抽的是下标 `num_tokens + j`。普通步是 decode-ahead：喂下标 `num_tokens` 的那个、抽 `num_tokens + 1`；
  投机轮之后的普通步（bonus 已计数）抽 `num_tokens`；prefill 后第一个抽 0。实现为 `GenerationBatch._draw_positions()`，
  投机批按 `_pending` 覆盖。
- **SEED-004** 草稿器与随机数：Qwen MTP 头在 fork 里恒贪心起草（`greedy_sampling=True`）；DFlash2 的候选选择器只在采样器有
  `sample_proposal` 时才抽样，否则 argmax——所以缺省组合（27B + dflash2）的起草不吃随机数。只有 DFlash 1（`qwen3_dflash`）
  拿采样器起草：包成 `_PositionedDraftSampler`，提议用目标同位置的 key（耦合抽样，发出的永远是目标的抽样，分布不变）。
  对 DFlash 1 接受率的影响**没量**（缺省不用它）。
- **SEED-005** DFlash 的自适应块长看草稿器上最近 8 轮的 `accept_lens`；fork 在新请求加入后第一轮 `reset` 草稿器，
  `qwen3_dflash.DFlashDraftModel.reset` 清空 `accept_lens/draft_lens`（DFlash2 继承它）。所以单独跑的请求块长序列从同一起点开始。
- **SEED-006** 变异检查：普通步位置 +1、投机批 `_draw_positions` 对非 pending 行 +1、MTP walk `base_positions` +1、DFlash walk
  `base_positions` +1，四种改法各至少让一个新测试失败。第一版测试沿用 spec-with-tools 的小世界（logits 5 对 -4），temp 1 下
  几乎总抽 argmax，前两种变异**没被抓到**；改用平坦的马尔可夫目标（`_flat_world_logits`）后才抓到。教训：测随机数接线的测试，
  分布必须平到换一把 key 就换一个 token。
- **SEED-007** 性能没量。每个 token 多一次 Python 里的 splitmix64 和一个 key 数组；`categorical` 的调用次数不变（投机 walk 对
  一块位置逐位调用，原来是一次整块——块长 ≤ 16，预期可忽略，但**没量**）。

## lmk 侧
- `sampling.parse_sampling`：`seed` 是 uint64，收 0…2^64−1，非整数 / 布尔 / 负数 / 越界回 400 点名 `seed` 并说出范围（第一轮曾收有符号范围，owner 改为 uint64，见 SEED-008）。
- 没带就由 `SeedSource.draw()` 抽 31 位（`secrets.randbelow`），`get/set_current_seed_source` 可注入（照 clock 的样子）。warmup 不抽。
- 回传：流式 usage chunk 的 `lmk.seed`；非流式应答新增 `lmk` 对象（与流式同一组字段）。
- `LmkChatDone`：`seed`、`seedFrom`（request / lmk）、`othersAtStart`、`othersPeak`；`sampling` 字段去掉 seed 不重复。
  `othersAtStart/Peak` 由准入队列维护（`Ticket`），数的是"已放行"的请求（含 warmup），不是引擎里的解码行；已放行但还在读 prompt 的也算。
  偏保守：可能把"其实没同批解码"的也算进去，不会漏算。

## 测试数
- 引擎：新文件 `tests/test_seeded_sampling.py` 24 个全过；不需要模型的引擎单测 331 个全过（排除需要真实模型/网络的文件，
  未跑全量——全量里有会去加载模型的测试，实验在跑、内存紧）。
- lmk：`make test` 233 passed、11 skipped（itest 跳过）；`make lint` 过。

## 集成测试（owner 跑；预期先写）
`test_itest_chat.py::test_the_same_seed_replays_the_same_answer_and_another_seed_does_not[plain|tools]`：
第一次不带 seed（冷，设 cache，拿回 lmk 抽的 seed）→ 同 seed 两次 → seed+1 一次。temp 1.0 / top_p 0.95 / top_k 20，max_tokens 160。

```bash
cd ~/src/lmk-seed
# 1) 不带草稿
LMK_ITEST=1 .venv/bin/python -m pytest -q -s tests/test_itest_chat.py -m itest -k same_seed
# 2) 带 dflash2（缺省组合；kv16）
LMK_ITEST=1 LMK_ITEST_DRAFT=$HOME/.cache/huggingface/hub/models--seabit-ai--Qwen3.8-27B-DFlash2-4bit/snapshots/local-quant-2026-09-24 \
  .venv/bin/python -m pytest -q -s tests/test_itest_chat.py -m itest -k same_seed
# 3) 合并前的全套（两种配置各一次）：去掉 -k same_seed
```
（`make itest` 在这个工作树里也行：它用 `~/src/lmk-seed/.venv` 与 `~/src/lmk-seed/.engine/mlx-engine`；但 Makefile 会先
`git fetch` 并把引擎工作树 checkout 到 `ENGINE_COMMIT`，HEAD 游离，之后 `git -C .engine/mlx-engine checkout lmk-seed`。）

预期（跑前写）：
- 同 seed 两次逐字相同：两种配置 × 两种形状都相同。把握：中高——路径已由单测锁住；风险在真实模型上我没看过的非确定性
  （GPU 归约在相同形状下应确定）。
- seed+1 与之不同：把握高（temp 1、160 token、思考段）。
- 冷算的第一次与恢复续跑的重放逐字相同：把握中。预期相同（owner：cache 只允许舍入级差别）；kv16 上 SPD-034 那种差别没量过。
  不同就打印分叉点——那是 cache 的线索，不是 seed 的失败。

## 第二轮（review 修正，2026-09-26）
- **SEED-008** 顺序路径（`ModelKit`，lmk 在 `max_parallel: 1` 且模型不是视觉套件时会走到）：`set_seed` 对负数抛错（请求变 500），
  且把 seed 截到低 32 位（s 与 s+2^32 撞）。修：seed 定为 uint64（owner）；lmk 只收 0..2^64−1，其余 400；`set_seed` 收整个范围，
  MLX 拿全值、NumPy / Python 拿 splitmix64 混合后的高 32 位；顺序路径的采样器也换成 `SeededSampler`（按调用顺序取位置）。
- **SEED-009** 提前结束的请求**不会离开引擎的批**：lmk 关掉 `create_generator` 返回的生成器，但 fork（与上游）的 `_batched_generation`
  没有 finally，从不调用 `remove`；只有 prefill 阶段取消与引擎自己匹配 stop 字符串两条路会移除。于是客户端断开或 lmk 在回答段
  匹配到 stop 之后，那一行继续解码到 EOS / max_tokens（lmk 不传 max_tokens 时引擎缺省一千万，实际到窗口为止），
  与之后放行的请求同批、拖慢它们，也让 `othersPeak` 说"独占"而实际不是。这不止是一个 tick：review 的预想（多一步）比实际轻。
  修：`_batched_generation` 捕获 `GeneratorExit` 调 `remove`；lmk 在 `finally` 里显式 `close()`。引擎的请求队列先来后到
  （`request_lifecycle.drain_generation_events`），所以移除一定排在之后放行的请求之前。**没在真机上量过**被丢下的行实际跑了多久。
  README "Closing the connection cancels the request" 的说法此前只对 prefill 阶段成立；现在对解码阶段也成立（单测，未真机验）。
- **SEED-010** 贪心不抽 seed：temp 0（或没有温度）时 `seedFrom: "greedy"`、`seed` 为 null，请求里带的 seed 也不传。
- **SEED-011** 投机开 / 关同 seed：itest 在短 prompt 上断言逐字相同（按位置取 key，logits 相同就相同）；长上下文不断言（SLC-006/010）。
- 测试：引擎 `test_seeded_sampling.py` 33 个，引擎不需要模型的单测 340 个全过；lmk 241 passed、12 skipped。

新增的集成测试命令（带草稿那条会多跑它）同上；只跑投机开关这一个：
```bash
LMK_ITEST=1 LMK_ITEST_DRAFT=$HOME/.cache/huggingface/hub/models--seabit-ai--Qwen3.8-27B-DFlash2-4bit/snapshots/local-quant-2026-09-24 \
  .venv/bin/python -m pytest -q -s tests/test_itest_chat.py -m itest -k "same_seed or draft_on_and_off"
```
预期：同上，另加投机开 / 关逐字相同（把握中高：短 prompt 上此前观察到逐字相同，但那是贪心；采样下平票更少碰到）。

## 第三轮：owner 跑 itest 后（2026-09-26 下午，exp01）
- **SEED-012** 不同路径（冷算 vs 恢复、投机开 vs 关）的分布差是 bf16 一格的量级：TV ≤ 0.10、p>0.05 的 token 上 |Δlogprob| ≤ 0.5；
  纯分块读 prompt 的对照 0.22–0.60（top-20）。采样在边界处会翻，所以这两种对比不逐字相同；itest 改为比分布。
- **SEED-013（bug，已修）** 带 DFlash 草稿器时，恢复前缀之后那段 prefill（以及紧接着的 decode 步，直到 RoPE 状态被别的调用设上）的
  RoPE 位置从 0 数起：首 token 分布 TV 0.24–0.30。fork `39c17a2`。修后冷算与恢复逐位相同。可能就是事故 000193（推测，验证办法见 exp01）。
  磁盘上修前写下的 cache 块已被污染，建议合并时 `CACHE_FORMAT_VERSION` +1（owner 定）。
- 修后 itest 全套两种配置都过：不带草稿 10 passed / 3 skipped，dflash2 12 passed / 1 skipped（`exp01-divergence-logits/raw/itest-full-after-fix.txt`）。
