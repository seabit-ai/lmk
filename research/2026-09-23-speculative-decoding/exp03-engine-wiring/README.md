# exp03：投机解码接进引擎的批处理路径——对不对、快多少

日期 2026-09-23。m3u。引擎 worktree `.engine/mlx-engine-spec` 分支 `lmk-spec`（上游 08f0c07 + KV 量化 + 投机解码），lmk 分支 `speculative-decoding`。
草稿器 = exp02 拆出的 MTP 头。方案见设计文档 §5。

## 怎么量
`smoke.py`：引擎的 `load_model`（批处理路径，`max_seq_nums=2`）+ `load_draft_model`，exp01 的三题，贪心 400 token：
无草稿 → 有草稿（sha 对照）→ 采样 temp 1.0 的接受率 → 两条请求同时（B=2）。`pair_probe`（scratch）用来隔离并发问题：
两条相同 prompt 对 code+story 混合。

## 预期（写在第一次跑之前的口头预期，事后补记；exp02 是依据）
- 贪心下 code / copyedit 与无草稿逐字节相同，story 分叉（exp02 同样如此）。把握：中高。
- 加速与 exp02 同量级（b=3：code 1.6×、copyedit 1.7×、story 1.2×）；引擎多一层账目，略低。把握：中。
- B=2 有收益但小于 B=1（mlx-vlm 的 Gemma 数据）。把握：低。

## 过程与结果（速度数全部在评测同时占 GPU 的条件下量，只有相对值有意义；干净的数见 results.md）
1. **第一次跑没有一轮投机**：Qwen3.5 家族每个 prompt 都带 mRoPE 的 `rope_deltas`（纯文本全 0），我把"有 rope_deltas"当成"有图片"退回普通解码。改成"非零才是图片"。
2. **代码第 33 个 token 分叉**。逐轮比目标模型 cache 与单步参考：第一轮后 KV 就差 8.6——位置编码不一致。我每轮清掉了 Qwen 用来接着数位置的
   `_position_ids/_rope_deltas` 侧状态（mlx-vlm 只在 prefill 前清一次）。改成和普通步一样同步 `rope_deltas`，验证前向自己调、把 `rope_deltas` 传进去。
   之后 **code 400 token 逐字节一致，copyedit 一致，story 的 sha 与 exp02 里 mlx-vlm 自己的循环完全相同（572686175c）**。
3. 贪心行改用 mlx-vlm 同款的融合 argmax 验证（`speculative_argmax_from_hidden`），引擎的 `create_sampler` 在 temp 0 时给 sampler 打 `greedy` 标记。
4. **采样开着（temp 1.0）的接受率：253/292 = 87%**（146 轮 400 token）——比想的高得多；目标 token 用行自己的 sampler 采，和贪心草稿比对。
5. **两条请求同时（B=2）**：两条相同 prompt → 两行都与 B=1 逐字节一致；code + story 混合 → code 第 53 个 token 分叉、story 第 157 个；
   把接受数统一成最小值反而让相同 prompt 也分叉。普通解码的 B=2 是对的。结论：mlx-vlm 的批量回滚在 Qwen 混合注意力 + 行长不齐时不对
   （per-row 的 cache roll 或 GDN 批量回滚），它自己的 README 只在 Gemma（无 GDN）上验证过批量。
   **MVP 收口：只在解码批里只有一行时投机**（`MAX_ROUND_ROWS = 1`），第二条请求进来就退回普通步，走了再恢复。批量账目留着，等查清再开。
6. 引擎全套单测与基线失败集一致；新增 8 个账目单测。

| 臂（GPU 争用下） | story | code | copyedit |
|---|---|---|---|
| 无草稿 | 22.2 tok/s · adad345a9c | 24.4 · 58b44530ab | 21.1 · 8fe1ef2be6 |
| 有草稿 | 27.7 · **572686175c（分叉，同 exp02）** · 1.25× | 32.7 · **相同** · 1.34× | 26.7 · **相同** · 1.27× |
