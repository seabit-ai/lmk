# exp01 — prefill 中途被取消，已经算过的部分还在不在缓存里

## 目的

后台预热（warmup）要做到"不打扰"：每算完一块检查一次，有真实请求在等就停下让路。
这条路成立的前提是：**停下时已经 prefill 过的那部分，下一次还能命中**。
如果只有整个请求正常结束才落盘，让路就等于白算，设计要换（比如预热请求本身只发一小段前缀）。

## 方法

- 服务：常驻的 lmk，模型 `qwen3.8-27b-4bit`（owner 机器上的正式服务，实验前 `lmk status` 确认无人在用）。
- 输入：合成文本，约 16k token 的 system（开头带一次性随机 nonce，保证没有任何缓存可命中）+ 一条短 user 消息。
- 步骤：
  1. **A**：流式请求（`max_tokens: 1`）。读流里的 `lmk.prefill` 进度块；当 `processed / total ≥ 0.5` 时直接关掉连接。
     记下关连接前最后一次看到的 `processed`、`total`、`cached`。lmk 的下一次进度写入会失败 → ClientGone → prefill 取消
     （`lmk/chat.py` on_prefill 返回 False）。
  2. 轮询 `/lmk/v1/status`，等 `in_flight` 为空（确认 A 已经收尾）。
  3. **B**：同一个请求体（非流式，`max_tokens: 1`），记下 `usage.prompt_tokens` 与 `prompt_tokens_details.cached_tokens`、耗时。
- 所有请求带 `X-Lmk-Purpose: research`，`X-Lmk-Ref-Id: prewarm-exp01-*`。

## 预期（运行前写）

两种结局，我的倾向略偏 H0（约 6 成）：检查点写盘（`persistcache.put`）是引擎在一次生成的收尾处做的，
我没有找到"prefill 进行中就落盘"的代码路径；取消走的是异常/提前返回，可能根本走不到收尾。

- **H1（前缀保住了）**：B 的 `cached_tokens` ≈ A 取消时的 `processed`（向下取整到检查点边界，256 或 2048 的倍数），
  即约 8k；B 只需再算后半段，耗时约为完整 prefill 的一半。
- **H0（白算了）**：B 的 `cached_tokens` ≈ 0（或只有几百 token 的模板开头），B 要从头算完整的约 16k。

另记：A 取消到 B 开始之间的收尾时长（status 里 in_flight 清空所需时间）。

## 对比基准

无（本调查第一个实验）。完整 prefill 时长由 B 在 H0 下的耗时给出；H1 下由 A 的速度外推。

## 运行

`./run.sh`（输出写到 `result.log`，原样保留）。

## 结果（2026-09-24 运行，原始输出见 `result.log`）

- A：total 14,583 token，每 2048 块约 6.3 秒（约 325 token/s）；在 `processed = 8192`（25.6 s）时关连接。
- A 收尾：status 里的 A 以 `cancelled` 结束，`total_ms = 38897`——关连接后又跑了约 13 秒。
- B：`prompt_tokens 14583`，**`cached_tokens 12288`**，耗时 14.5 s。

**H1 成立，与我运行前的倾向（H0）相反。** 取消时已经算完的块留在了缓存里：12,288 = 6 × 2048，正好是 A 实际停下的位置
（不是关连接时的 8,192——见下一条）。

**意外：断开检测滞后两块。** 客户端关连接后，lmk 下一次写进度块进了内核 socket 缓冲区，不报错；再下一次写才得到
EPIPE/RST → ClientGone → prefill 取消。所以客户端断开到 prefill 停下差了约两块（这里约 13 秒）。
