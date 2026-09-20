# exp02 — 模型写出来的工具调用原文，与 mlx-lm 自带解析器

日期 2026-09-19。

## 目的
lmk 最大的一块未知工作量：引擎是 token 进、文本出，**不**解析工具调用（LMS-014）。mlx-lm 自带
`tool_parsers/qwen3_coder.py`。要知道：27B 实际写出的文本是什么样；这个解析器能不能把它变成
**类型正确**的 JSON——XML 风格的参数全是字符串，整数 / 布尔 / 数组要靠工具的 schema 还原。
结果决定这块活是"接上现成的"还是"自己写"。

## 方法
- 同 exp01：借 LM Studio 自带的 Python 与库，直接驱动引擎，27B-4bit。这次带上了 `request_id`。
- 用模型自己的聊天模板渲染 `messages + tools`（`apply_chat_template(..., tools=…)`），渲染结果存
  `rendered-prompt.txt`。两个工具，故意覆盖三种非字符串类型：`file_read(path, max_lines:integer)`、
  `grep_search(pattern, paths:array<string>, case_sensitive:boolean)`。
- 一句话任务同时需要两个工具：读 notes.md 前 20 行 + 在 docs 和 src 里大小写不敏感地搜 teapot。
- `max_tokens=700, temp=0`。原始输出存 `raw-output.txt`；取 `</think>` 之后的部分，按解析器声明的
  起止标记切块，逐块交给 `qwen3_coder.parse_tool_call(block, TOOLS)`。

## 预期（运行前写下）
1. 输出 = 思考 + `</think>` + 两个 `<tool_call>…</tool_call>` 块（并行）。把握中高——经 LM Studio 时
   它就是并行发的（exp02 / exp07）。
2. 块内格式是 `<function=name>` + 若干 `<parameter=key>value</parameter>` + `</function>`，与聊天模板里
   教的一致。把握高。
3. 解析器两块都能解析，且类型还原正确：`max_lines` 是整数 20（不是 "20"）、`case_sensitive` 是布尔
   false、`paths` 是数组 ["docs","src"]。把握中——数组那一项最悬：模型可能写成 JSON 数组、也可能
   写成逗号分隔的文本，解析器对后者未必认。
4. 解析器的返回形状能直接拼成 OpenAI 的 `tool_calls`（name + arguments）。把握中高。

## 对比基准
经 LM Studio 的 exp02（结构化结果已知：两个并行调用，参数合法）。

## 运行
    ./run.sh

## 结果（2026-09-19，`result.json` / `raw-output.txt` / `rendered-prompt.txt` / `stderr.log` 原样保留）
四条预期全中。
1. ✓ 思考 + `</think>` + 两个并行的 `<tool_call>` 块。
2. ✓ 块内格式与聊天模板教的一致：`<function=…>`、每个参数一个 `<parameter=key>` 块、值独占一行。
3. ✓ 类型还原全对：`max_lines` → 整数 `20`；`case_sensitive` → 布尔 `false`；`paths` → 数组
   `["docs","src"]`（模型写的就是 JSON 数组字面量，解析器按 schema 的 `array` 类型解开）。
4. ✓ 返回 `{"name": …, "arguments": {…}}`，直接就是 OpenAI `tool_calls[].function` 的内容
   （`arguments` 再 `json.dumps` 一次即可）。
带上 `request_id` 后，exp01 里那条"请带 request_id"的告警消失了。

## 读数
- **lmk 的工具调用这一块是"接上现成的"，不是"自己写"**：渲染用模型自带的聊天模板，解析用 mlx-lm
  自带的 `qwen3_coder`（Apple 从 Qwen 官方解析器改的），类型还原靠请求里的工具 schema。
- lmk 自己要写的只剩：流式地把输出切成 思考 / 正文 / 工具调用块 三路，以及给每个调用生成 id。
- N=1，两个工具，最简单的类型。没测：嵌套对象参数、参数值里含 `</parameter>` 字样、模型写坏格式时
  的表现（引擎的 `tool_runtime` 语法约束能不能兜住）、别的模型族。
