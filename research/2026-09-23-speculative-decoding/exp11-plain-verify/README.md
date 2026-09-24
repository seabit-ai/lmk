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

## 过程与结果
（跑完补）
