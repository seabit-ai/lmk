# exp12：DFlash 2 接进引擎的批处理路径——对不对、快多少

日期 2026-09-24。m3u。fork 分支 `lmk-dflash`（从 25d1c38 起）：`Drafter` 两种（mtp / dflash）、`dflash_round`、prompt 分块抓 5 层隐状态只留窗口尾巴、
校验走普通前向 + 记录回滚（exp11）。草稿器 `incoai/Qwen3.8-27B-DFlash2`（bf16，block 8，窗口 2048）。

## 怎么量
exp05 的 `smoke.py` 改草稿器路径：三题贪心 400 token 无/有草稿（sha 对照）、temp 1.0 接受率、双请求；kv16 与 kv8 各跑一次（`KV_BITS`）。

## 预期（跑之前写）
- E1 不崩；三题 sha 与普通解码一致（exp07 在 mlx-vlm 自己的循环里三题全一致；引擎路径的校验现在也是普通前向）。把握：中。
- E2 速度：一轮 ≈ 草稿 17.5 + 校验 T≈5–8 约 45 ms；每轮接受 code 4.3、copyedit 4.9、story 2.0 ⇒ code ≈ 1.8×（约 70 tok/s）、copyedit ≈ 2×、story ≈ 0.9–1.0×。把握：中低。
- E3 temp 1.0 的 code 接受率 ≈ 80%（exp07）。把握：中。
- E4 kv8 同 kv16（校验路径对量化 cache 不用钩子：普通前向的注意力自己吃 "causal" 字符串 mask）。把握：中。

## 过程与结果（`raw/smoke-kv16.txt`、`raw/smoke-kv8.txt`；fork 3b493b5）
1. 第一次跑在兼容性检查上停了：DFlash2 的 config `model_type` 是 "qwen3"（它训练的目标家族），种类要从 `architectures`（`DFlash2DraftModel`）/ `dflash_config` 判断。修了。
2. 结果（贪心 400 token）：

| | 普通 | DFlash2（引擎路径） | 倍数 | MTP 头（exp11，同路径） |
|---|---|---|---|---|
| kv16 story | 39.5 | 40.0，分叉 | 1.01× | 1.20×，分叉 |
| kv16 code | 39.4 | 59.4，**一致** | **1.51×** | 1.53×，一致 |
| kv16 copyedit | 39.3 | 72.0，**一致** | **1.83×** | 1.51×，一致 |
| kv8 story / code / copyedit | 38.8 / 38.6 / 38.5 | 38.1 分叉 / 59.6 **分叉** / 70.9 一致 | 0.98× / 1.54× / 1.84× | 1.15× 分叉 / 1.53× 一致 / 1.51× 一致 |
| 接受 / 草稿（kv16，三题合计） | | 845/1212 = 70%，486 轮 | | 764/1129 = 68%，566 轮 |
| 采样 temp 1.0 code 接受率 | | 81%（kv16）/ 78%（kv8） | | 89% |
| 双请求（投机不开） | | 64.6 tok/s 合计 | | 59.8 |

- E1 半命中：不崩；code / copyedit 一致（kv16），story 分叉（每轮接受 2 左右时草稿几乎全拒，分叉源自校验的浮点平手）；kv8 的 code 也分叉（sha 4590fb15b9，与 0.6.16 exact 下的相同——
  量化 cache 上 L>1 的注意力与 L=1 不位级一致，MTP 头在 kv8 上 code 一致是运气）。
- E2 部分命中：copyedit 1.83×（预期 2×），code 1.51×（预期 1.8×），story 1.0×（预期 0.9–1.0）。轮数比 MTP 少 14%（486 对 566）但每轮更贵。
- E3 命中：81%。E4 命中：kv8 不崩、速度同 kv16。
3. **在 M3 Ultra 上，DFlash2 与 MTP 头打平**：code 一样（1.51 对 1.53），copyedit 赢（1.83 对 1.51），story 输（1.01 对 1.20），草稿器 4 GB 对 810 MB。
   27B 页的缺省先仍是 `mtp`，`model.draft: dflash2` 可选；M4 Pro 一类的机器上 DFlash2 应更占优（草稿一步的固定开销摊薄），等 `lmk bench` 的行。
