# exp15：两条请求同时投机——今天的校验路径上，exp03 的批量回滚错误还在不在

日期 2026-09-24。m3u。fork `lmk` 3b493b5。起因：owner 让重看"投机只在单行"（`MAX_ROUND_ROWS = 1`）。exp03（0.6.12，mlx-vlm 的 target_verify
+ 它自带的 gdn_states 回滚）里 code + story 同时投机，code 第 53 个 token 分叉。今天的路径不同：普通前向校验 + `GdnVerifyRecorder` 记录的输入 →
`rollback_speculative_cache` 的"重跑前缀"分支（批量时按每行 `valid_ends` 重跑）。

## 怎么量（`smoke.py`，MTP 头；DFlash2 的轮目前只写了单行，MTP 结果好再补）
普通解码三题做基准；单独投机三题；四组两两同时（code+story、code+code、code+copyedit、story+copyedit），每条输出对照**同一 prompt 单独普通解码**的 sha，
记每条 tok/s、合计 tok/s、投机轮数。`ROWS=1`（今天的行为，对照）与 `ROWS=2` 各跑一次；kv16。

## 预期（跑之前写）
- E1 ROWS=2 下 code / copyedit 在两两同时时仍与单独普通解码一致（回滚按行重跑前缀，行长不齐不再是问题）。把握：中——exp03 的错在 mlx-vlm 批量回滚，
  重跑分支是另一段代码，但 mask / 左填充 / 每行 offset 都可能再出错。
- E2 story 分叉可以接受（单独投机也分叉）；关键看 code 是否与单独时一样一致。
- E3 合计速度：ROWS=1 约 64 tok/s（exp12 双请求）；ROWS=2 约 75–85（1.2–1.35×）。宽 2×(1+草稿) 的校验比两行各一步贵，收益比单行小。把握：低。
- 判据：E1 成立且 E3 ≥ 1.15× ⇒ 打开两行投机；E1 不成立 ⇒ 记下分叉点，保持单行。

## 过程与结果
（跑完补）
