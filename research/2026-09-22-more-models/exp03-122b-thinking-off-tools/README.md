# exp03：122B 关掉思考后，工具调用还对不对

日期 2026-09-22。m3u，Qwen3.5-122B-A10B-4bit，`model.thinking: false`（模板渲染空的 think 块），temp 0。常驻服务停掉。

## 预期（跑之前写）
1. 集成测试（`LMK_ITEST_THINKING=off make itest`）**5/5**：工具调用、cache、warmup、图片、跨重启。把握：中高——
   Qwen 的工具调用格式由模板决定，不依赖思考；但没了思考，`max_lines` 这类参数类型（整数而非字符串）可能出错。
2. 三个 agent 形状的任务（`tasks.sh`，raw/）：
   - T1 两步：读文件 → 根据内容改文件。预期第一轮只发 file_read，第二轮发 file_edit 且 old_text 取自工具结果。把握：中。
   - T2 不该用工具的问题（"what does 'idempotent' mean"）：预期无工具调用、直接回答。把握：高。
   - T3 参数类型：读 notes.md 前 5 行 → `max_lines: 5` 是整数。把握：中。

## 结果（2026-09-22，raw/）
1. ✓ 集成测试 **5/5**（130 s）。
2. ✓ 三个任务全对（raw/summary.txt）：
   | 任务 | 结果 | 输出 token |
   |---|---|---|
   | T1 步 1 | 只发 `file_read {path: notes.md}` | 26 |
   | T1 步 2 | `file_edit`，old_text `- buy oat milk` 逐字取自工具结果，new_text `- buy almond milk` | 57 |
   | T2 | 无工具调用，一句话正确解释 | 32 |
   | T3 | `file_read {path, max_lines: 5}`，`max_lines` 是 int | 38 |
   每步几十个 token、一秒内——与思考开时"几千 token 起步"是两个世界。
条件与未测：单次、temp 0、四个简单任务；长会话、多工具、需要推理的任务没测。
