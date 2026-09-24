# exp11：校验改普通批量前向（记录 GDN 输入给回滚）——MTP 头先量，DFlash2 随后

日期 2026-09-24。m3u。fork 分支 `lmk-plainverify`（25d1c38，从 `lmk` bc9588e 起）。裁决：own-engine.md 09-24 补记（三）。
改动：`_verify_block` 不再 `speculative_verify=True`（0.6.16 的 exact verifier），改成普通前向；`GdnVerifyRecorder` 在前向期间钩住
`language.gated_delta_update` 和标记过的 GDN conv，记下 (q,k,v,a,b,A_log,dt_bias,初始状态,mask,卷积输入,K,None) 交给 mlx-vlm 的
`rollback_speculative_cache`（无中间态 → 重跑接受前缀的分支）。

## 怎么量
exp03 的 `smoke.py`（kv16 + MTP 头）与 exp05 的（kv8 + MTP 头），输出在各自的 `raw-plainverify/`：贪心三题 sha 对普通解码、tok/s、采样接受率、双请求。
对照：0.6.12（exp03/05 原始）与 0.6.16 exact verifier（`raw-vlm616/`）。

## 预期（跑之前写）
- E1 不崩、回滚正确：接受不满时状态回到接受位置——若回滚错，后续 token 会明显乱（sha 分叉且文本可见地坏）。把握：中高。
- E2 速度回到 0.6.12 的水平或更好：kv16 code ≥ 1.45×（一轮 = 草稿 ~10 + 校验 T=4 ~43 ms，2.7/轮 ⇒ ≈ 51 tok/s；0.6.12 是 58，它的校验有专用内核）。把握：中。
- E3 一致性退回 0.6.12 水平：code / copyedit 一致，story 可能分叉。把握：中高。
- E4 kv8 同上；kv8 + 草稿的 code 在 0.6.16 exact 下是分叉的，普通前向下可能回到一致（0.6.12 时一致）。把握：低。

## 过程与结果（`exp03-engine-wiring/raw-plainverify/`、`exp05-kv8-with-draft/raw-plainverify/`）

| 贪心 400 token | 0.6.12（原） | 0.6.16 exact verifier | **0.6.16 + 普通前向校验** |
|---|---|---|---|
| kv16 story / code / copyedit | 1.14× 分叉 / 1.49× 一致 / 1.48× 一致 | 0.92× 一致 / 1.26× 一致 / 1.24× 一致 | **1.20× 分叉 / 1.53× 一致 / 1.51× 一致** |
| kv16 code tok/s | 57.7 | 49.3 | **60.6** |
| kv8 story / code / copyedit | 1.14× 分叉 / 1.49× 一致 / 1.48× 一致 | 0.96× 分叉 / 1.23× **分叉** / 1.29× 一致 | **1.15× 分叉 / 1.53× 一致 / 1.51× 一致** |
| 轮数 · 接受/草稿（kv8） | 582 · 748/1163 | 586 · 744/1171 | 582 · 748/1163（与 0.6.12 完全相同） |
| 采样接受率 code | 86–87% | 85% | 89% / 82% |

- E1 命中：回滚正确（接受不满时状态回到接受位置；文本正常、轮数与 0.6.12 一模一样）。
- E2 超预期：code 60.6 tok/s，比 0.6.12 的 57.7 还快一点（普通批量前向的注意力是一次 SDPA，0.6.12 的 target_verify 是逐位置循环）。
- E3 命中：一致性回到 0.6.12 的水平——kv16 story 的 sha 572686175c 与 exp02 里 mlx-vlm 自己循环的完全相同。
- E4 命中（把握低的那条）：kv8 + 草稿的 code 回到一致。
结论：这条校验路径进 `lmk` 分支；DFlash2 接线走同一条。
