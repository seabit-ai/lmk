# 带 tools 的请求用上投机解码：投机轮逐位过 logits processor

> 2026-09-25 裁决（owner "y"）。取证：`research/2026-09-25-spec-with-tools/notes.md`（SPD-022..031）。

## 起因
带 `tools` 的请求会挂上引擎的工具守卫 logits processor；`speculative.py` 的 `_can_round` 见到任何 processor 就不走投机轮。
owner 机器上 113 个 agent 步只有 1 个起草（SPD-023）。bench 与模型页的速度是在不带 tools 的请求上量的。

## 裁决：A——llama.cpp 式逐位 walk（owner "y"）
投机轮里对带 processor 的行按位置顺序：先让 processor 处理该位置的 logits → 抽样（或贪心取 argmax）→ 与草稿比对 → 不一致即停。
processor 只吃被接受的 token，不需要回滚 processor 状态（SPD-027）。之后结构化输出的 `json_schema` 约束直接复用（其设计文档裁决六的 backlog 项由此落地）。
- 先修 SPD-028：投机轮 → 普通步交接时 bonus token 被 processor 吃两次（调用体内 llguidance 抛 ValueError）。
- **否决 B（只在正文段起草，进调用体退回普通步）**：调用体是接受率最高的一段（113 个 turn 里 91 个以工具调用收尾），且结构化输出还得再做一遍。
- **否决 C（vLLM 式整块掩码 + rollback）**：好处在多行并行，而多行投机仍不对（SPD-021）；多一套试探再撤回的状态。

## 验收（照 CLAUDE.md 09-25 教训：覆盖主用法形状、按类别拆开报）
- 引擎单测：同一 prompt + processor，逐 token 与投机轮产出的 token 序列相同；覆盖思考段、正文、调用体、调用后收尾四段。
- itest：带 tools 的 agent prompt，贪心 400 token，开/关草稿 sha 逐字节一致；报告按段（思考 / 正文 / 调用体）的 drafted 与 accepted。
- 真机：kitten 真实会话若干轮，`LmkChatDone` 的 turn 请求 `draftDrafted>0`；按 purpose 拆开报。
- 模型页 Known issues 那条在验收过后删除，速度表补"带 tools 的 agent 请求"一行。
