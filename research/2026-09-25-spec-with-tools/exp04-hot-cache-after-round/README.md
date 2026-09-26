# exp04：一轮投机收尾的请求交出的热 cache，比它报的 all_tokens 少一个 token 吗——下一轮带 tools 的对话会不会命中它、输出变不变

日期 2026-09-25。m3u。review 提出的 I1（自 b58a72e 起就有，e053a72 让带 tools 的 agent 轮也走投机后会碰到）：行在轮里结束时，轮的最后一个 token
按设计不进 cache，但 `_response` / `_emit_pending` 交出的 `all_tokens` 含它；下一请求整段前缀命中热 cache（trim 0）时，引擎以为 len(all_tokens) 个都在 cache 里，
实际 offset 少一。

## 怎么量（`run.py`；`ENGINE` 环境变量指引擎目录）
27B-4bit，kv8，思考开 + low，dflash2（推荐组合）。走 lmk 的 `run_chat`（真实渲染、切分、工具调用解析），草稿开关按请求注入。
1. 第一轮：带 tools 的 agent 请求（exp01 的 fizzbuzz），**开草稿**，贪心。结束后读引擎的热 entry：`len(prompt_input_ids)` 对 cache 里 KV 层的 offset。
2. 第二轮 prompt = 第一轮 + assistant（reasoning_content + tool_calls）+ tool 结果，用 lmk 的模板渲染后 tokenize：它是不是以热 entry 的 token 整段开头（整段命中的前提）。
3. 第二轮三种条件，都关草稿、贪心、`top_logprobs=5`、60 token：A 热 cache 原样（旧引擎 = bug 条件）；B 先清掉热 cache（走磁盘快照，诚实）；
   记首位置 top-5 logprob 与全文。旧引擎 e053a72 跑一次（`raw/old/`），修复后跑一次（`raw/fixed/`）。

## 预期（跑之前写）
- E1 旧引擎：热 entry 的 token 数 = KV offset + 1。把握：高（读码 + 单测已红）。
- E2 Qwen 模板重渲染第一轮时与生成的 token 逐个相同，第二轮 prompt 以热 entry 整段开头 ⇒ 整段命中、trim 0。把握：中（思考段、工具调用的重渲染空白可能与生成的不同；
  不同就不会命中热 cache，bug 在这条路上不触发，只剩磁盘快照——磁盘快照是按 256 块、用诚实的长度存的）。
- E3 若 E2 成立：A 与 B 首位置 logprob 差明显大于 exp02 那种 1 ulp（缺一个 `<|im_end|>` 且后面位置整体错一位），文本可能不同。把握：中低。
- E4 修复后：热 entry 的 token 数 = KV offset；A 与 B 的差回到浮点噪声量级。把握：中。

## 过程与结果（`raw/old-run.txt` + `raw/old/results.json`：引擎 e053a72；`raw/fixed-run.txt` + `raw/fixed/results.json`：修复后 42a248c）

| | 旧 e053a72 | 修复后 42a248c |
|---|---|---|
| 第一轮（开草稿）| file_write，drafted 234 / accepted 224 | 同左 |
| 热 entry 的 token 数 / KV offset | **807 / 806**（末尾 `<\|im_end\|>` 不在 KV 里） | 806 / 806 |
| 第二轮 prompt 与热 entry 的公共前缀 | 807 = 整段（trim 0） | 806 = 整段（trim 0） |
| 第二轮引擎报的 cached_tokens | 807（多报一个） | 806，未命中 26 个从 `<\|im_end\|>` 起重算 |
| 第二轮 A（热）首位置 top-5 logprob | Now −0.125 · The −2.25 · Let −4.75 · Good −5.625 · File −5.875 | Now −0.125 · The −2.5 · Let −4.875 · Good −5.375 · File −6.125 |
| 第二轮 60 token 全文 | "Now let me run it. … run_command python fizzbuzz.py" | 相同 |

- **E1 命中**：旧引擎交出的热 cache 比 all_tokens 少一个 token（807 对 806），缺的正是这一轮的 `<|im_end|>`。
- **E2 命中**：Qwen 模板重渲染第一轮（reasoning_content + tool_calls）与生成的 token 逐个相同，第二轮整段命中热 cache、trim 0——bug 在 agent 对话的主路径上触发。
- **E3 落空（方向对，幅度小）**：直接比"旧 A"与"修复后 A"（bug 的净效应）：首位置 logprob 差 ≤ 0.25，60 token 文本相同。缺一个 `<|im_end|>`、后面位置整体挪一位，
  在这一例里没把输出带偏。另：A 对 B（热对磁盘）在修复前后都差到 2.75–3.0——**这个对照不能判 bug**：修复后热 cache 诚实了，A/B 仍差这么多，
  说明 kv8 下"解码/校验时写进的 KV"与"磁盘恢复 512 + 重新 prefill 320"本身就给出明显不同的 logits（与 exp01–03 冷/热不一致同源，未查）。
- **E4 部分命中**：热 entry token 数 = offset，命中数诚实；A/B 的差没有回到噪声量级（见上）。
