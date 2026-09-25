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

## 过程与结果（`raw/smoke-rows1.txt`、`raw/smoke-rows2.txt`）

| 两条同时，MTP 头，kv16，贪心 400 token | ROWS=1（今天） | ROWS=2 |
|---|---|---|
| code + story：每条 / 合计 | 30.7 + 31.0 / 59.2，code 一致 | 36.1 + 29.9 / 57.2，**code 分叉** |
| code + code | 30.5 + 31.4 / 58.8，都一致 | 36.9 + 36.8 / 68.4，**一条分叉** |
| code + copyedit | 42.4 + 31.2 / 54.4，都一致 | 48.9 + 35.6 / 57.7，都一致 |
| story + copyedit | 41.0 + 31.2 / 50.4 | 36.7 + 35.4 / 48.3 |

（ROWS=1 的 code+copyedit、story+copyedit 有 98–120 轮投机：短的那条答完后，剩下的一条单行投机。）
- **E1 落空**：两条相同的 code 同时投机，一条分叉；同一配对不投机时两条都一致 ⇒ 错在两行一起的校验/回滚，不在批量解码。exp03 的问题在新校验路径上仍在。没追分叉点。
- **E3 落空，比预期还差**：最好 1.16×（code+code），其余持平或更慢；两行同时时校验变宽的代价吃掉了收益。
- 判据不满足 ⇒ 保持 `MAX_ROUND_ROWS = 1`。即使修好分叉，上限约 1.16×，不值得；DFlash2 不做多行。
