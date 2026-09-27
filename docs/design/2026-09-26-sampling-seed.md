# 每次生成都有一个已知的 seed：采样的答案可以重放

> 2026-09-26 owner 批准计划（"go"）。实现：分支 `sampling-seed`（lmk）+ fork 分支 `lmk-seed`（引擎）。取证与实现记录：
> `research/2026-09-26-sampling-seed/notes.md`（SEED-001..）。**取代** `2026-09-21-sampling.md` §3（"seed——无法兑现，如实报"）。

## 起因
kitten 从 lmk 拿到一个坏答案（qwen3.8-27b-4bit，temp 1.0 / top_p 0.95 / top_k 20，dflash2 投机，kv16，max_parallel 2），复现不出来：
采样是随机的，而 lmk 把 `seed` 丢了（`sampling.IGNORED_KEYS`，因为引擎批处理路径上 "Seed arg is ignored for batched gen"）。
事故不能重放，就只能猜。

## 裁决：事故可重放 = 同一请求 + 同一 seed，单独跑
1. **引擎（fork）按请求吃 seed，按位置取随机数。** 带 seed 的请求拿到 `SeededSampler`：生成序列第 p 个 token 的抽样用
   `key(seed, p)`（splitmix64 混合后 `mx.random.key`），是 (seed, p) 的纯函数，不是一条随用随推的随机流。于是：
   - 一行的随机数与批里别的行无关（全局 PRNG 与别的请求都挪不动它）。
   - 普通步、prefill 后第一个 token、MTP / DFlash 的校验 walk、带 processor 的逐位 walk、DFlash 草稿器的采样提议，
     对同一位置用同一把 key（mlx-vlm 已有的 `sample_target(logprobs, row_ids, positions)` 协议）。logits 相同时，
     投机与不投机抽出同样的 token；草稿提议与目标共用 key（耦合抽样），发出的 token 永远是目标的抽样，分布不变。
   - 不知道位置的调用方（mlx-lm 的批生成器，每个发出的 token 调一次；顺序路径 `ModelKit`）按调用顺序拿 0, 1, 2, …。
     顺序路径的全局 `set_seed` 也收整个 uint64：MLX 拿全值，NumPy / Python 拿 64 位混合后的 32 位（原来截到低 32 位，s 与 s+2^32 撞，负数抛错）。
   - `seed=None` 保持原来的无 seed 采样器；temp 0 仍是贪心，seed 不起作用。
   - 顺带修：消费方提前关掉生成器（客户端断开、lmk 在回答段匹配到 stop）时，fork 的 `_batched_generation` 现在把这一行移出批；
     原来这一行会一直解码到 EOS / max_tokens，和之后放行的请求同批（见下"批的组成"）。
2. **lmk：每次采样生成都有一个已知 seed。** seed 是 **uint64**（owner，2026-09-26）：请求带 0..2^64−1 的整数就用它；负数、更大、非整数回 400，
   点名 `seed` 并说出范围。不带，lmk 自己抽一个（31 位，`SeedSource` 可注入，测试换成计数器）。
   **贪心不抽**：temp 0（或没有任何温度，即引擎缺省贪心）时没有随机数，`seed` 为 null、`seedFrom: "greedy"`，请求里带的 seed 也不传不回。
   warmup 不抽（它只读 prompt，不采样）。
   - 回给调用方：流式的 usage chunk 与非流式的应答里 `lmk.seed`（非流式应答此前没有 `lmk` 字段，这次一并加上，含 `restore_ms` 等）。
   - `LmkChatDone` 记 `seed`、`seedFrom`（`request` / `lmk` / `greedy`），以及 **`othersAtStart` / `othersPeak`**：这个请求被放行时、以及在引擎里的
     任何时刻，另有几个请求（含 warmup）也在引擎里。`othersPeak == 0` = 它独占引擎，重放条件满足。
     数的是准入队列的放行名单。它可信的前提：一个请求离开名单（`admission.leave`）时，它在引擎里的行已被移除或移除已排队。
     正常结束时引擎在发出最后一个 token 的那一步就移除了行；提前结束时 lmk 在离开名单之前关掉引擎的生成器（`MlxEngine.generate`
     的 `finally`），fork 把移除请求放进引擎的请求队列；引擎按先来后到处理这个队列，所以之后放行的请求插入之前那一行已经不在了。
     `sampling` 字段不再重复 seed。
   - `LmkParamIgnored` 机制保留（`IGNORED_KEYS` 现为空）。

## 能保证什么，不能保证什么
- **保证（单测锁住，itest 验）**：同一请求、同一 seed、单独跑 → 逐 token 相同；另一个 seed → 另一个答案。itest 在同一 cache
  状态下断言逐字相同；冷算对恢复续跑只打印（预期也相同，见下一条之后的 cache 一节）。
- **批的组成是条件。** 两个请求同时解码时，引擎走多行普通步（投机只在一行时跑，SPD-021），多行的矩阵运算与单行的舍入不同；
  随机数不受影响，但 logits 的舍入差可能在近乎平票的位置翻转一次抽样。所以重放要单独跑，日志里的 `othersPeak` 告诉你原来那次是不是单独跑的。
  原来那次不是单独跑的，重放出同一答案的概率仍高（翻转只发生在平票处），但不是逐字保证。
- **prefix cache 不该影响结果（owner，2026-09-26）。** 冷算、热 cache 续跑、磁盘恢复续跑，预期给出相同的 logits，差别只在浮点舍入
  （不同的分块形状让归约顺序不同）。比这大的差别——例如 SPD-034 的首位置 logprob 差 2.75——**是 bug，要修**，另行跟踪，不是可接受的限制。
  seed 重放正是抓这类 bug 的工具：同 seed 下冷算与恢复续跑分叉，就是一条线索。
- **投机开 / 关，同 seed：** 抽样按位置取 key，所以只要 logits 相同，开关投机抽出同样的 token。短 prompt 上 logits 逐位相同
  （模型页："on short prompts code and copy-editing matched exactly"），itest 在短 prompt 上断言逐字相同；长上下文里块校验的舍入与
  逐 token 不同，平票处可能翻转（SLC-006/010），那里不断言。
- 投机解码下 DFlash 的自适应块长取决于最近几轮的接受情况；这段历史在一个新请求加入时清零（fork 的 `_round` 在行数变化时 reset 草稿器），
  所以单独跑的重放块长序列相同。

## 被否的方案
- **A. 引擎全局 `mx.random.seed(seed)`（原 2026-09-21 §3 的第一版裁决）。** 批处理里所有行共用一条全局随机流：别的行抽一次，
  这一行的随机数就挪一位——正是要消灭的依赖。也被引擎自己在批处理路径上丢掉了。
- **B. 每行一条随机流（`key = split(key)` 每次调用推一步）。** 行之间独立了，但一个位置用哪把 key 取决于之前调用了几次：
  投机轮对整块位置抽样、被拒的草稿位置也抽过，普通步有 decode-ahead 的多抽，于是投机开关、块长、何时退回普通步都会改变
  后面每一个 token 的随机数。按位置取 key 没有这个问题，且与 mlx-vlm 的 positioned-sampler 协议一致。
- **C. 只在 lmk 侧记录、不改引擎（例如记下 logits 或全部 token 做事后比对）。** 能看到"答了什么"，不能回答"同样的输入会不会再这样答"，
  也就无法区分采样的偶然与 cache/批处理的 bug。
- **D. 不带 seed 就不抽（保持 None）。** 事故发生时恰恰是没带 seed 的请求（kitten 不发 seed），那就永远重放不了。
  抽一个的代价是零；因此每次生成都有 seed。

## 验收
- 引擎单测（`tests/test_seeded_sampling.py`，24 个）：同 seed 同抽样、异 seed 异；位置决定 key、与调用历史和全局 PRNG 无关；
  一行 token 独跑与夹在两行中间相同；普通步第 k 个 token 用第 k 把 key；投机轮（MTP / DFlash × 有无 processor × 3 个 seed）
  与普通路径逐 token 相同，且轮里既有接受也有拒绝；投机轮之间插普通步也相同；DFlash 草稿提议用目标同位置的 key。
  变异检查：把任一处位置 +1，都有测试失败。
- 引擎单测（第二轮）：顺序路径收 0..2^64−1、s 与 s+2^32 不撞、重放不受全局 PRNG 影响；mlx-lm `BatchGenerator`（极小的随机 llama，
  不下载模型）上带 seed 的一行独跑与夹在两行中间相同；提前关生成器移除这一行、读完不移除。
- lmk 单测：seed 的校验（uint64 以外 400）与透传（含 2^64−1）、抽取与回传、贪心不抽、日志字段、warmup 不抽、`othersAtStart/othersPeak`、
  提前结束关掉引擎生成器。
- itest（`LMK_ITEST=1`，`test_the_same_seed_replays_the_same_answer_and_another_seed_does_not`，普通与带 tools 两种）：
  不带 seed 的第一次（冷，设 cache）→ 拿回的 seed 重放两次，逐字相同 → seed+1 不同。冷算与恢复续跑是否相同只打印不断言
  （预期相同；不同就是 cache 的线索，见上）。要在不带草稿、带 dflash2 两种配置各跑一次。
- itest `test_the_same_seed_draws_the_same_tokens_with_the_draft_on_and_off`（只在带草稿时跑）：短 prompt，同 seed，逐请求关投机 / 开投机，断言逐字相同。
