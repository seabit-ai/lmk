# exp01：上下文自检的开销与"恢复后不打分"

起因（owner 跑的 itest，`../raw/itest2.log`，27B-4bit、kv16）：
- (a) 不带草稿、磁盘恢复 768 后 prefill 367：`scored_tokens=0`；带 dflash2 的同一路径打了 368 个。
- (b) `promptSurpriseMs` 约 4.5–6 s / 1k 打分 token（冷 799 token 打 512 个用了 3.1 s）。

## 预期（跑前写）
- (b) 的主因是计时把**这一块 prefill 的 forward 本身**算进去了：`add()` 里 `.tolist()` 触发惰性求值，hidden 还没算，
  于是整块 forward 在计时区里跑完。把握：中高（3.1 s 对 799 token ≈ 这台机器冷 prefill 的量级）。
  次因候选：`speculative_logits_from_hidden` 走 `_EXACT_SPECULATIVE_VERIFIER.quantized_linear`，可能比普通 `lm_head` 慢。把握：低。
  修正后纯打分开销预期 0.1–0.3 s / 512 行（owner 的估计），即 < 该段 prefill 的 10%。
- (a) 原因未知，先复现再说。候选：恢复路径上某一块的 forward 没把 `return_hidden` 传到语言模型，或 hidden 的长度与块长不一致被 `add()` 跳过。

## 做法
`run.py`：直接用 `MlxEngine`（不经 HTTP），同 itest 的历史 + 新一轮；分别计 prefill 墙钟、打分 ms；给 SurpriseMeter 打日志看每块收到的 hidden 形状。

## 结果（2026-09-26，m3u，27B-4bit，kv16，不带草稿；`raw/run1..4`）
- **(a) 找到了**：恢复后的纯文本 prefill 走 fork `patches/qwen3_5.py` 的快路径 `_patched_vlm_qwen3_5_language_model_call`
  （text_only 且没有 position_ids），它不认 `return_hidden`，`hidden_states=None`（`raw/run2-probe.txt`：这条路上根本没进 mlx-vlm 的
  `LanguageModel.__call__`）。冷 prefill 带着 position_ids 走原路径，所以有。带 dflash2 时 `capture_layer_ids` 让它不算 text_only，也走原路径。
  修：快路径在要求时返回 `[hidden]`。**顺带**：MTP 草稿器在恢复后的文本请求上第一轮也拿不到状态（原来要等一个普通步），同一处修好。
- **(b) 两个原因都在，预期的主因猜对了一半**：
  1. 计时包含这一块的 forward：冷 787 token，修前 `ms` 3075，墙钟 3141——几乎全是 prefill（`raw/run1.txt`）。修：打分前先 `mx.eval(rows)`，计时只包投影。
  2. `speculative_logits_from_hidden` 走 mlx-vlm 的 exact verifier：512 行 **545–585 ms**；普通 `lm_head` **78 ms**（`raw/run3-fixed.txt`）。
     预期里我把它标成"把握低"的次因——猜错在这一侧：它是 7 倍。修：没有 softcap 的模型直接用 `lm_head` / 绑定的 embedding；分 128 行一块算（float32 一行 1 MB）。
- **修后开销**（`raw/run4-lm-head.txt`）：冷 787 token 打 512 个 **87 ms**，占这次墙钟 2655 ms 的 3.3%；磁盘恢复 768 后段 325 打 324 个
  **56 ms**，占 1338 ms 的 4.2%。约 0.17 ms / 打分 token。都在 10% 以内，`TAIL_TOKENS` 保持 512。
  条件：单请求、无并发、段 ≤ 1k；更长的冷 prefill 打分仍是 512 行，占比只会更小。
