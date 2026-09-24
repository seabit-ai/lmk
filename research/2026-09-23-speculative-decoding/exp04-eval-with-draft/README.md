# exp04：草稿开着，评测分掉不掉（投机解码验收的第二条）

日期 2026-09-23。27B-4bit，`reasoning_effort: low`，`speculative_decoding: true`，lmk 默认采样（模型的 generation_config：temp 1.0），
智能评测的原题（code 前 20、instruct 10、tools 5，各 2 遍，无前缀），临时 lmk 端口 1236，常驻停。对照 = 09-23 上午的 27B-low 臂
（code 99%、instruct 100%、tools 100%）。

## 预期（跑之前写）
- 贪心下代码逐字节相同已证；这里是采样，草稿只改变"采到哪个 token"的随机路径，不改分布 ⇒ 分数与对照同一水平：每类相差 ≤1 题。把握：中高。
- 采样接受率约 70%（exp03 smoke 69%）。把握：中。
- 判据：每类不比对照少 2 题及以上 ⇒ 通过，27B-4bit 页写推荐 `speculative_decoding: true`。

## 第一遍（并发 2）的问题
harness 默认两条请求并行，而投机只在单请求解码时跑（exp03 第 5 条），70 题只跑了 349 轮——大部分 token 是普通解码出来的，
这一臂测的不是"草稿开着"的分数。结果留在 `raw/draft-on/`（code 39/40、instruct 20/20、tools 10/10，12 s/答）。
第二遍 `raw/draft-on-solo/`：`EVAL_PARALLEL=1`（harness 新加的开关），每题单流。
