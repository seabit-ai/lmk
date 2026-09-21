# `lmk up` 打印哪些 agent 的配置片段

日期 2026-09-21。起因：公开前检查——`lmk up` 只打印 kitten 的片段，陌生用户不知道 kitten 是什么。
owner："we need to make it generic. for example, what does openclaw config looks like?"；另：kitten 迟早公开，名字可以出现。

- **ACS-001 OpenClaw 自定义 provider 的形状。** 出处两个，互相对得上：① OpenClaw 2026.6.9 安装包自带文档
  `docs/gateway/local-models.md` 的 "Other OpenAI-compatible local proxies" 一节；② m3u 上在用的 `~/.openclaw/openclaw.json`
  里 `lmstudio` / `omlx` 两个本地 provider。要点：`models.providers.<id>` 下 `baseUrl`（**带 `/v1`**）、`apiKey`（任意非空）、
  `api: "openai-completions"`（lmk 没有 `/v1/responses`）、`models[]`（`id` 不带 provider 前缀；选用时写 `lmk/<id>`）；
  `models.mode: "merge"` 保留云端模型；文件是 JSON5。图片输入要 `input: ["text", "image"]`。
- **ACS-002 片段里的值从哪来。** `id` / `contextWindow` / `input` 取自 `/lmk/v1/status` 的 `model`（真值，不是模板里的占位数）。
  `reasoning: false`、`cost` 全 0 照抄官方模板。`maxTokens` 官方模板是 8192；lmk 把思考也算进 `max_tokens`，而实测一步写过
  15,698 token（THK-004），8192 会截断，故取 32768。
- **ACS-003 验证到哪。** 渲染出的片段用 OpenClaw 自带的 json5 包解析通过。**没在真 OpenClaw 里跑过**：m3u 上 `openclaw agent --help`
  本身就挂住（>2 分钟无输出），另起隔离 profile 需要另一套 gateway，没做。`reasoning: false` 对思考模型在 OpenClaw 里的实际效果未知。
