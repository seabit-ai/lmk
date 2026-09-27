# 上下文自检：prefill 时顺手量模型对自己 prompt 的"意外程度"

> 2026-09-26 owner "yes"。实现：lmk 分支 `context-check`（基于 `sampling-seed`）+ fork 分支 `context-check`（基于 `lmk-seed`）。
> 取证与待测：`research/2026-09-26-context-check/notes.md`（CTX-001..）。

## 起因
事故 000193：qwen3.8-27b-4bit（线性注意力与全注意力混合）从磁盘恢复 83712 个 token、prefill 1186 个，思考里说"这只是个
system prompt，用户什么都没问"——像是把 85k 的对话整个丢了。引擎日志无异常；同一条恢复路径重放三次都正常。
假设：恢复或 prefill 出来的上下文**有时**是坏的。下次再发生，日志里要有证据。

## 2026-09-26 实测后的修订（待 owner 裁）
exp03 / exp04（`research/2026-09-26-context-check/`）推翻了原裁决 1–2 的前提：
- **对整段 prompt 打分不是健康信号，而是反相关的。** 指令微调模型对 user / system / 工具文本没有被训练去预测：同一段数数文本，
  不套模板 surprise 0.02，放进 user 消息 6.28（几乎每个位置都押 `<|im_end|>`）。把恢复出的状态清零，它忘了"自己在一条 user 消息里"，反而"预测得更好"。
  owner 第一次 itest 里"破坏后更低"（2.05 → 1.29）就是这个。
- **只打 assistant 段（模型自己写过的回合）分得开**：历史里有、段里要复述的内容，intact 0.09–0.11，状态全清零 1.30–1.34，
  换成另一段对话的状态 1.11（约 10 倍）。只清递归状态看不出来（0.10：全注意力层靠 KV 就能抄）。
- **修订**：lmk 按家族的回合标记（Qwen：`<|im_start|>assistant` … `<|im_end|>`，含末尾的生成提示）找出 prompt 里 assistant 段的
  token，把下标（`check_targets`）交给引擎，引擎打其中落在未缓存段里的最近 512 个；没有已知标记的家族（Gemma 4 未量、PLAIN）
  或带图请求退回原来的"段尾 512"，日志 `promptSurpriseScored` 写 `targets` / `tail`。段里没有 assistant 段时打分为 0 个——
  这时没有信号，不假装有。
- **开销实测**（exp01，m3u 27B-4bit）：原实现把这一块 prefill 的 forward 算进了计时，而且投影走 mlx-vlm 的 exact verifier
  （512 行 545–585 ms，普通 lm_head 78 ms）。修后约 175 ms / 1k 打分 token，约为该段 prefill 的 3–4%。
- 顺带修掉一个真 bug（exp01）：恢复后的纯文本 prefill 走 fork 的快路径，丢了 `return_hidden`——自检在所有磁盘/热恢复的请求上
  都打不到分，MTP 草稿器在这类请求的第一轮也拿不到状态。

## 原裁决（2026-09-26 上午）
1. **量什么**：prefill 未缓存段时，对段里的真实 token 做 teacher-forced 打分——位置 t 的最终层状态经 lm_head 投影，取 token t+1 的
   log 概率。上下文完好，模型能预测自己的对话；上下文丢了或坏了，它对自己的对话"意外"。记 **surprise = −logprob**（nats/token）：
   均值、p90、最大值、打了多少个 token、花了多少毫秒。
2. **只打段尾 512 个位置**（`TAIL_TOKENS`）。状态是 forward 本来就算的（只在够得着段尾的那几块 prefill 上多要一份 `return_hidden`），
   额外开销是每个位置一行 lm_head 投影加一次 logsumexp。512 是上限不是实测最优；每个请求的 `promptSurpriseMs` 就是实测开销，
   按"每 1k 个打分 token 多少毫秒"写进 notes（CTX-003）。段的第一个 token 不打（预测它的是恢复前缀最后一个位置的状态，手里没有）；
   段的最后一个位置预测的是第一个生成 token，不打。
3. **记在哪**：引擎每个请求留一份记录（恢复来源 none / hot / disk、恢复了多少 token、hot 恢复截掉多少、段长、上面的统计），
   lmk 在生成结束时取走，写进 `LmkChatDone` 与 `LmkWarmupDone`：`restoreSource`、`restoredTokens`、`hotTrimmedTokens`、
   `promptSurpriseMean/P90/Max/Tokens/Ms`、`promptSurpriseUnsupported`（模型没有从状态投影到 logits 的方法时说明为什么没打分）。
4. **报警**：`LmkContextSurprising`（WARN）在均值高于 `CONTEXT_SURPRISE_WARN_MEAN` 且打分 ≥ 32 个 token 时发出。**阈值现在是 None
   ——不报警，只记数。** 正常区间要从真机日志里量出来（按 purpose、按恢复来源拆开），再定阈值；不猜一个数。
5. **测试钩子**：fork `model_kit.RESTORED_CACHE_HOOK`，生产里是 None；itest 用它把恢复出来的线性注意力（递归）状态清零，
   证明这个检查看得见"上下文丢了"。

## 被否的方案
- **E. 对整段 prompt（或段尾）打分**（原裁决）：见上方修订——在指令微调模型上与上下文好坏反相关。
- **A. 用 prefill 已经算出的 logits。** qwen3_5 的 prefill 对整块算 lm_head（没有 `logits_to_keep`），但 MLX 是惰性的：
  不用就不算。拿来用就要为整块（最多 2048 × 248k）算投影，比只算段尾 512 行贵几倍、峰值内存大得多。
- **B. 事后另跑一遍打分请求。** 事故是"有时"发生，重跑拿到的是另一次恢复的状态，正是要抓的那次拿不到。
- **C. 只记恢复来源和 checkpoint、不打分。** 000193 的引擎日志已经说明恢复"看起来正常"；缺的是"状态本身对不对"的量。
- **D. 比对恢复状态与重算状态（逐层数值对比）。** 最直接，但每次都要重算整个前缀，83k token 要几十秒，只能做成调试工具，不能常开。

## 局限（写明）
- surprise 取决于内容：工具返回的文件内容本来就难预测，正常请求的值也会分散。所以阈值要按量出来的分布定，可能要按 purpose 分。
- 只打段尾：段很短（例如续一轮对话只有几十 token）时样本少；< 32 个不判。
- 视觉 token 所在位置（图片占位）也会被打分；带图请求的数不可与纯文本请求直接比较。
- 模型没有 `speculative_logits_from_hidden` 时不打分（Qwen3.5 家族与 Gemma 4 有）。
