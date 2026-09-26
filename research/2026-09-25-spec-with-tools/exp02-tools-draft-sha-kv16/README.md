# exp02：exp01 换 kv16——分叉是 kv8 的浮点平手，还是 walk 的错

日期 2026-09-25。m3u。同 exp01（fork e053a72，同一 prompt、同一 `run.py`，放在 `../exp01-tools-draft-sha/run.py`），只改 `KV_BITS=16`。

## 为什么有这个实验
exp01（kv8）：贪心开草稿与关草稿在思考段第 67 个字符分叉（关：`fizzbuzz(n)`，开：`` `fizzbuzz(n)` ``）；而且**关草稿的冷跑与热跑（prompt cache 命中 512）
之间就已经不一致**——kv8 上 prefill 走的路径不同，输出就不同。exp12 记过 kv8 的 L>1 注意力与 L=1 不位级一致。要把"量化浮点"与"walk 逻辑错"分开，
在不量化的 cache 上再跑一次。

## 预期（跑之前写）
- E1 kv16：关草稿冷跑与热跑一致。把握：中（exp12 kv16 下 code/copyedit 一致，但那是无 cache 命中的比较）。
- E2 kv16：开草稿与关草稿（同为热跑）sha 一致。把握：中——walk 在单测里与逐 token 路径逐位相同；剩下的风险是 exp12 story 类那种
  "校验前向 L>1 与 L=1 浮点平手"（kv16 下也发生过，story 类）。本 prompt 的思考段是散文，有这风险。
- 若 E2 落空：看分叉处的 logits 是否是平手（前两名差极小）；是则归为 exp11/12 已接受的浮点平手，否则是 walk 的错，停手报告。

## 过程与结果（`raw/run.txt`、`raw/results.json`、`raw/margin.txt`）
- 关草稿冷 e6be828b4fc7（297 token）/ 热 2d3df3d3a2dc（299）——**E1 落空**：kv16 上冷热也不一致。
- 开草稿 ×2：f9e1be8e5119，66.0 tok/s，按段计数与 exp01 kv8 **逐项相同**（sha 也与 kv8 开草稿相同，见"未解释"）。与关·热在思考段第 210 字符分叉。**E2 落空**。
- 分叉处 logits（`margin.py`：一次无 cache 的整段前向，prompt + 共同前缀）：`'\n\n'` 24.125、`'\n'` 24.000——bf16 在 16–32 之间的间距正是 0.125，
  **一个 ulp 的平手**。关草稿选了 `\n\n`（接着再想一句），开草稿选了 `\n`（接 `</think>`）。归为 exp11/12 已接受的浮点平手，不是 walk 的错
  （walk 在这个位置只对 `<tool_call>` 施掩码，改不了 `\n` 与 `\n\n` 的先后）。
- 采样：drafted 274 / accepted 235；思考 78 / 64（82%），调用体 192 / 170（89%）。

**未解释**：开草稿的输出在 kv8 与 kv16 上逐字节相同（本 prompt 与 exp03 的 prompt 都是），计数也相同；关草稿在两种 kv 上不同。要么轮里的校验/回滚
没有用到量化 cache 的差异（值得查：推荐组合是 kv8），要么关草稿路径本身在这些 prompt 上不稳定（冷/热就不同）。没查，交给 review。
