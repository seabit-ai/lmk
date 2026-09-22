# exp05：Gemma 4 26B-A4B——lmk 的第二个模型家族

日期 2026-09-22。m3u，权重 `~/models/gemma-4-26b-a4b-it-4bit`（oMLX 留下的，15 GB；HF `mlx-community/gemma-4-26b-a4b-it-4bit`）。
lmk 分支 `gemma4`（Dialect 落地后）。常驻（122B-48GB）测试期间停。

## 事实（写代码前查的）
- 模板：思考**缺省关**，`enable_thinking: true` 在首个 system turn 顶部放 `<|think|>`；思考内容是 `<|channel>thought\n…<channel|>`，
  思考关时模板替模型写一个空 channel。工具调用 `<|tool_call>call:name{k:<|"|>v<|"|>}<tool_call|>`（mlx-lm 有 gemma4 解析器）。
  工具结果要 `tool_responses: [{name, response}]` 挂在 `role: tool` 的消息上，之后模型**在同一个 turn 里**接着答（模板不再开 model turn）。
- generation_config：temp 1.0 / top_k 64 / top_p 0.95。MoE 128 专家、激活 4B；vision 有；窗口 262k。

## 预期（跑之前写）
1. 集成测试，思考关（缺省）：**5/5**。把握：中——第一次跑非 Qwen 家族，隐含假设会被翻出来；最可能出问题的是工具往返
   （tool_responses 形状、同 turn 续写）与图片（`<|image|>` 占位）。
2. 集成测试，思考开（`LMK_ITEST_THINKING=on`）：5/5，且 `reasoning` 非空。把握：中低——模型自己开 channel，切分器的 leading 阶段要认
   `<|channel>thought\n`。
3. decode：激活 4B、4bit ⇒ 每 token 读约 2 GB ⇒ **100+ tok/s**。把握：中。prefill：≈ 1000+。把握：低。
4. 三个 agent 任务（exp03 同款，思考关）3/3。把握：中。

## 结果（2026-09-22，raw/）
| | 预期 | 实得 |
|---|---|---|
| 集成测试，思考关（缺省） | 5/5 | **5/5**，44 s ✓ |
| 集成测试，思考开 | 5/5 且有思考 | **5/5**，但工具那条**没有思考内容**——见 F1 |
| 加载 / 窗口 | — | 6 s；baseline 14.5 GiB；262k 保住 |
| decode | 100+ | **113.5 tok/s** ✓ |
| 冷 prefill | ≈1000+ | **约 1,850 tok/s**（4,026 uncached，2.18 s；bench 一开始把它误判为"seed 复用"，见 F2） |
| cached prefill | — | 87k tok/s，首 token 0.26 s |
| 三个 agent 任务（思考关） | 3/3 | **3/3**（21 / 43 / 27 token；T2 262 token，见 F3） |

翻出来的隐含假设（这正是第二个家族的价值）：
- **F1 "思考开 ⇒ 每轮都有思考"是 Qwen 的巧合。** Qwen 的模板把 `<think>` 硬塞进生成提示；Gemma 只在 system 顶部放 `<|think|>`，
  开不开 channel 是模型每轮自己决定的——读文件这种显然的一步它直接发工具调用（28 token）。⇒ Dialect 加 `thinks_every_turn`，
  集成测试按它断言。
- **F2 "冷 = 命中为零"是 Qwen 的巧合。** Gemma 的 cache 能把 `<bos><|turn>user\n` 这种每条相同的头（10 token）取回来。
  ⇒ bench 改为"命中不足一个 256 块仍算冷"。
- **F3 Gemma 的 `thinking: false` 是提示不是硬开关。** 模板替模型关了一个空 channel，模型仍可能自己再开一个：T2（一句话解释
  idempotent）思考了 240 token 才答 19 个词。切分器正确切出（reasoning_content 里），但"关思考"对 Gemma 不保证零思考。
  另外三个任务没思考。
- 消息形状：`role: tool` + `tool_responses`，模型在同一 turn 续写——集成测试的工具往返通过，即这个形状对。

## 修正（同日）：上面的数来自 oMLX 留下的旧修订，不是客户会拿到的
`~/models/gemma-4-26b-a4b-it-4bit` 与 HF 当前 commit `0d77464`（2026-07-05）逐文件对照：三个权重分片、config、chat_template、
tokenizer_config 全不同——是更早的修订。按客户路径 `lmk pull`（459 s，15.6 GB）重测，以下为准（raw-hf-0d77464/）：
| | 旧修订 | **HF 0d77464** |
|---|---|---|
| 集成测试 关 / 开 | 5/5 / 5/5 | **5/5 / 5/5** |
| 冷 prefill | ~1,850 | **1,833 tok/s** |
| cached prefill | 87k | 83k |
| decode | 113.5 | **119.5 tok/s** |
| agent 任务 | 3/3 | **3/3**（T1 步 1 思考了 100 token 才调工具——又是 F3） |

- **F4 新模板换了工具结果的形状。** 0d77464 的模板原生认 OpenAI 的 `role: tool` + `tool_call_id`（自己按前一条 assistant 的
  tool_calls 回填函数名），把 `tool_responses` 标为 "Legacy (Google/Gemma native)"；我按旧模板做的转换在新模板上导致渲染异常
  （`str + None`）。⇒ Dialect 分 `GEMMA4`（原样传）与 `GEMMA4_LEGACY_TOOLS`（转 tool_responses），按模板文本里有没有
  `tool_call_id` 选。新模板还会把 assistant 消息上的 `reasoning_content` 渲染回 thought channel（lmk 目前不传，未用）。
- 教训：**进表的模型必须是 `lmk pull` 拿到的那份**，本机现成的副本只能当线索。
