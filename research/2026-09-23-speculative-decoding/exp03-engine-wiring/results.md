# exp03 结果（2026-09-23，GPU 空闲时量）

`raw/results.json`（smoke）、`raw/bench-spec.txt`（临时 lmk 开 `speculative_decoding: true` 的 `lmk bench`）。27B-4bit，草稿器 = 自带 MTP 头，每轮猜 3。

| 贪心 400 token | 无草稿 | 有草稿 | 输出 |
|---|---|---|---|
| story | 39.3 tok/s | 46.2 tok/s（**1.18×**） | 分叉（与 mlx-vlm 自己的循环同一结果 572686175c） |
| code | 39.2 | 57.9（**1.48×**） | **逐字节相同** |
| copyedit | 39.1 | 57.2（**1.46×**） | **逐字节相同** |
| code，temp 1.0（Qwen 默认采样） | — | 49.9（1.27×） | 接受 232/334 = 69% |
| 两条请求同时（code + story） | — | 合计 58.4 tok/s | 退回普通解码（并发限制见 README 第 5 条） |

- 与 exp02（mlx-vlm 原生 B=1 循环，b3）比：code 1.59× → 1.48×，copyedit 1.70× → 1.46×，story 1.22× → 1.18×。引擎多一层账目 + 批处理路径的
  草稿器不做整段 prompt 的预填（从空 cache 起步），少了约一成。
- 集成测试带草稿器：思考开 4/4、关 4/4（`LMK_ITEST_DRAFT`）。
- 验收：代码/复述逐 token 一致 ✔；散文分叉但评测分不降——评测臂未跑（27B-low 评测在草稿开着时的分数），**写模型页推荐前要补**。
