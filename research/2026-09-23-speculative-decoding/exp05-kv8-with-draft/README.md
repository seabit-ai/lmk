# exp05：投机解码 + 8 位 KV cache 一起开——先崩，修，再验

日期 2026-09-24。m3u（96 GB）。引擎 fork 分支 `lmk`（2839cfa 起），lmk 分支 `config-example`。
起因：owner 在自己的机器上按 27B 页的推荐把 `kv_cache_bits: 8` 和 `speculative_decoding: true` 同时打开，第一个请求 500。
两个功能各自验收过（exp02 长上下文评测、exp03/04），**合在一起从没跑过**——而推荐配置正是合在一起。

## 现场（`raw/service-crash-traceback.txt`，常驻服务的 stderr）
`mlx_vlm/models/qwen3_5/language.py:1641  prefix_len = keys.shape[-2] - L  → AttributeError: 'tuple' object has no attribute 'shape'`。
链路：投机轮的校验前向（`speculative.py:_verify_block`，一次喂 1 + block_size 个 token，`target_verify=True`）→ mlx-vlm 注意力
`update_and_fetch` 在量化 cache 上返回 `(packed, scales, biases)` 元组 → `_target_verify_left_padded_attention` 见到 `cache.bits` 直接返回 None
→ 退到逐位置循环，那里按稠密 keys 切片。普通解码 L=1 不进这条支路，所以两个功能各开一个都好。
结论：**上游 mlx-vlm 不支持 MTP 校验 + 量化 KV**，不是我们接线的错，但也没人替我们兜。

## 修法（fork 一个 commit）
校验块的逐位置循环等价于一次带因果掩码的整块注意力：块内第 i 个 token 看前缀 + 块内 0..i。量化 cache 时改走一次
`quantized_scaled_dot_product_attention`（它本来就吃 mask），批量 cache 的左填充位置也掩掉。挂在 fork 的 `patches/qwen3_5.py`，
替换 mlx-vlm 模块级的 `_target_verify_left_padded_attention`：只在"量化 cache 且 L > 1"时接手，稠密路径一字不动（exp03 的逐字节一致不受影响）。
单测：随机 q/k/v，量化 cache 存 300 个前缀 + 4 个新 token，整块掩码注意力 对 "反量化后逐位置循环" 的参考，容差 1e-3；带左填充的两行批量 cache 同样对照。

## 预期（跑之前写）
- E1 引擎级冒烟（`smoke.py`，`load_model(kv_bits=8)` + 草稿器）不打补丁必崩、打了不崩。把握：高。
- E2 打补丁后，贪心 400 token：code / copyedit 与 **kv8 普通解码** 逐字节一致（同一套量化 keys，只是分块算 vs 逐 token 算，差别只在浮点求和顺序）；
  story 分叉（稠密下 exp03 也分叉）。把握：中高（70%）。猜错的一侧：整块 `quantized_matmul` 的累加顺序和逐 token 不同，可能在几十 token 后翻一个 argmax。
- E3 采样 temp 1.0 的接受率 ≈ exp03 的 87%（±5）。把握：中。
- E4 加速与 16 位同量级（kv8 普通解码 38.9 tok/s 对 16 位 39.5，`docs/benchmarks.md`）：code ≈ 1.5×。把握：中。
- E5 lmk 集成测试两个开关同开（`LMK_ITEST_KV_BITS=8 LMK_ITEST_DRAFT=…`）5/5，重启还原。把握：中高。

## 过程与结果（`raw/smoke.txt`、`raw/results.json`；引擎 7a1e17f）
1. 补丁前的崩溃：常驻服务的 traceback（`raw/service-crash-traceback.txt`）；补丁后冒烟整套跑完，无异常。**E1 命中。**
2. 单测：整块掩码量化注意力对"反量化后逐位置循环"的参考，最大差 1.3e-7（float32；两行带左填充的批量 cache 同样）。
   第一版测试两条都挂，是测试自己的错：`quantized_scaled_dot_product_attention` 里 `queries *= scale` 是**原地**乘，先算被测再算参考，参考拿到的是缩放过的 q。
3. 贪心 400 token（kv8 普通解码 → kv8 + 草稿）：

| | story | code | copyedit |
|---|---|---|---|
| kv8 普通解码 | 38.8 tok/s，sha eb9ef544c2 | 38.7，58b44530ab | 38.6，8fe1ef2be6 |
| kv8 + 草稿 | 44.2（1.14×），bc4c30b9b8 **分叉** | 57.7（**1.49×**），**一致** | 57.1（**1.48×**），**一致** |

   **E2 命中**：code / copyedit 逐字节一致，story 分叉（16 位也分叉，SPD-007）。code 的 sha 58b44530ab 与 exp03 16 位下的完全相同——
   这个 prompt 上 8 位量化没改变贪心输出。
4. 采样 temp 1.0，code：接受 252/294 = **86%**，147 轮，55.9 tok/s。exp03（16 位）253/292 = 87%。**E3 命中。**
5. 双请求（story + code 同时，投机只在单行解码时开）：code 与 B=1 普通解码一致；story 的 sha b058517e64 与 exp03 双请求下的 story **完全相同**。
   合计 58.3 tok/s。与 exp03 第 5 条一致，没有新问题。
6. lmk 集成测试 `LMK_ITEST_KV_BITS=8 LMK_ITEST_DRAFT=…`：5/5（含重启后 cache 还原）。**E5 命中。** 71 s。
7. owner 的机器上按用户的样子打开两个开关：`lmk status` 显示 `KV cache 8-bit · speculative decoding on`，两个代码请求接受率 84%，0 失败。
8. E4 命中：code 1.49× 对 16 位的 1.5×。

## 教训（进 CLAUDE.md）
推荐配置是**组合**，验收也得按组合跑：两个开关各自 5/5，合起来第一个请求就 500。exp02 与 exp03/04 各测各的，模型页却把两条写成一段推荐。
