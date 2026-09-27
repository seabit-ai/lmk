# 上下文自检：取证与待测（CTX）

设计在 `docs/design/2026-09-26-context-check.md`。分支：lmk `context-check`（基于 `sampling-seed`），fork `context-check`（基于 `lmk-seed`）。
都未 push、未合。`ENGINE_COMMIT` 指向 fork `context-check` 的 commit。

## 发现
- **CTX-001** 引擎原来不告诉外面前缀是从哪来的：`RestoredPromptCache` 只有长度。hot 恢复还可能截掉热 cache 的尾巴
  （`_load_hot_restore_plan` 的 `trim_count`）；含线性注意力状态（`ArraysCache`）的模型不可截，截不了就退回磁盘。现在记 `source`
  与 `trimmed_tokens`。事故 000193 的 83712 = 327 × 256，是磁盘恢复落在 256 边界上的样子（与 LMK-002 一致）。
- **CTX-002** qwen3_5 的 prefill 块调用会构建整块的 lm_head 输出，但不被求值（MLX 惰性），所以"直接用已有 logits"并不免费（设计否决 A）。
  最终层状态 `out` 是每块本来就算的；`return_hidden=True` 只是把它多挂进返回值。与 DFlash 的 `capture_layer_ids` 同时要时，
  最终层排在 `hidden_states` 最后，检查把它取走，草稿器看到的仍只有它要的那几层（单测锁住）。
- **CTX-003 开销：没量。** 实验在跑，没加载模型、没跑 GPU 微基准。估算（未验证）：27B 的 lm_head 为 5120 × 248k，4 位量化；
  512 行投影约 0.65 T 乘加（512 × 5120 × 248k），同样 512 个 token 的 prefill 约 14 T（512 × 27B），约为其 1/20；预期每请求几十毫秒。实测数来自两处：
  itest 打印 "ms per 1k scored tokens"；上线后每个请求的 `promptSurpriseMs`。量到后写在这里，超过每请求 100 ms 就把 `TAIL_TOKENS` 降下来并说明。
- **CTX-004 正常区间：没量。** `CONTEXT_SURPRISE_WARN_MEAN = None`，只记数。定阈值的做法：合并部署后收一周 `LmkChatDone`，
  按 purpose × restoreSource 看 `promptSurpriseMean` 分布，再加 000193 类事故（若再发生）的值，写成实验。

## 集成测试（第一版，已被 exp04 的设计取代；下面的预期留作记录）
`tests/test_itest_chat.py::test_the_context_check_sees_a_corrupted_restore`：约 600 token 的历史（26 个代号 + 周次），冷算一次；
新一轮把同一张表再列一遍并要求复述（intact：恢复历史后 prefill 新一轮）；再来一轮，恢复后用 `RESTORED_CACHE_HOOK` 把所有
`ArraysCache`（线性注意力的递归状态）清零（corrupt）。

```bash
cd ~/src/lmk-seed
# 不带草稿（context-check 分支；Makefile 会把引擎工作树 checkout 到 ENGINE_COMMIT，之后 git -C .engine/mlx-engine checkout context-check）
LMK_ITEST=1 .venv/bin/python -m pytest -q -s tests/test_itest_chat.py -m itest -k context_check
# 带 dflash2（检查与 DFlash 的层捕获同时要状态）
LMK_ITEST=1 LMK_ITEST_DRAFT=$HOME/.cache/huggingface/hub/models--seabit-ai--Qwen3.8-27B-DFlash2-4bit/snapshots/local-quant-2026-09-24 \
  .venv/bin/python -m pytest -q -s tests/test_itest_chat.py -m itest -k context_check
# 合并前全套：去掉 -k
```
预期（跑前写）：
- intact：`restore_source` 为 disk（混合模型的热 cache 不可截，退回磁盘 256 边界），打分 ≥ 32 个，surprise 均值 < 2 nats
  （新一轮大半是把上文的表再抄一遍，完好的上下文预测得很准）。把握：中——"< 2" 是松的健全性边界，不是量出来的区间。
- corrupt：均值比 intact 高 1 nat 以上。把握：中。全注意力层（约四分之一）的 KV 还在，模型仍可能靠它抄表，差距可能小于预期；
  若如此，改为同时打乱 KV 再测（新实验，不覆盖）。
- 每 1k 打分 token 的毫秒数：几十到一百多毫秒（见 CTX-003），把握低。

## 2026-09-26 下午：owner 跑过 itest 之后（exp01–exp04，模型已可加载）
- **CTX-005**（exp01）恢复后不打分：fork 的 qwen3_5 纯文本快路径丢 `return_hidden`。已修。
- **CTX-006**（exp01）开销：计时含 forward + exact verifier 投影慢 7 倍。修后 ≈ 175 ms / 1k 打分 token，段 prefill 的 3–4%。CTX-003 的估计（1/20）方向对，
  itest 量到的 4.5–6 s / 1k 是测量错误。
- **CTX-007**（exp03）恢复出的状态与冷算一致（surprise 均值 5.01 对 4.92；恢复 1024 时 2.41 对 2.48）；逐位差大是不同 nonce 的位置错位，两次冷算之间同样大。
- **CTX-008**（exp03）指令微调模型对 user 文本的逐位预测：不套模板 0.02，套进 user 消息 6.28，argmax 几乎处处是 `<|im_end|>`。
- **CTX-009**（exp04）assistant 段分得开：intact 0.09–0.11 / 全清零 1.30–1.34 / 错的对话 1.11；只清递归状态 0.10（看不见）。
- **CTX-004 更新**：正常区间仍待真机日志；现在的量只对 assistant 段有意义，阈值也按它定。

集成测试命令不变（`-k context_check`）；新的 itest 是 `test_the_context_check_sees_a_lost_or_wrong_context`，两种配置都过（`exp04-by-role/raw/itest-*.txt`）。
