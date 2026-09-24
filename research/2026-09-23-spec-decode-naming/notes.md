# 别家怎么给"投机解码 / 草稿模型"起名（给 lmk 用户面定名用）

日期 2026-09-23。起因：投机解码 SAD 第 2 点，owner "yes, please do research first"。来源：各家官方文档/CLI help 原文（网页抓取）、本机 mlx-lm 0.31.3 / mlx-vlm 0.6.12 源码。

| 框架 | 草稿从哪来、怎么指 | 每轮猜几个 | 默认 | 统计 |
|---|---|---|---|---|
| llama.cpp | `--spec-draft-model` / `-md` / `--model-draft FNAME`；方法 `--spec-type draft-simple / draft-eagle3 / draft-mtp / draft-dflash / ngram-*`（2026-05 起 MTP 进主线，`--spec-type mtp`，不用另下模型） | `--spec-draft-n-max`（默认 3）、`--spec-draft-n-min`、`--spec-draft-p-min` | 关（"default: unused"） | — |
| LM Studio | GUI "Speculative Decoding" 区的 "Draft Model" 下拉（Power User 模式），要求"same vocabulary"；SDK `draftModel: <key>` | 未文档化 | 关 | `acceptedDraftTokensCount` / `predictedTokensCount` |
| Ollama | 草稿属于模型（Modelfile），不属于服务器；`draft_num_predict`（外挂草稿默认 4；**内置 MTP 张量要显式设**；0 = 关）；MLX runner 上 Gemma 4 MTP 后来默认开 | `draft_num_predict` | 关（Gemma 4 MLX 例外） | — |
| mlx-lm | `--draft-model` | `--num-draft-tokens`（默认 2） | 关 | 每 token `from_draft` |
| mlx-vlm | `--draft-model`，`--draft-kind dflash / eagle3 / mtp`（按草稿器 model_type 自动识别） | `--draft-block-size`（默认取草稿器 config 的 `block_size`，Qwen MTP = 3） | 关 | `accept_lens`、`speculative_stats_since` |
| vLLM | `--speculative-config '{"method": draft_model / ngram / suffix / mtp / eagle3 / dflash, "model": …, "num_speculative_tokens": N}'` | `num_speculative_tokens` | 关 | — |
| SGLang | `--speculative-algorithm EAGLE / EAGLE3 / NEXTN / STANDALONE / NGRAM / DFLASH`，`--speculative-draft-model-path` | `--speculative-num-steps`、`--speculative-eagle-topk`、`--speculative-num-draft-tokens`（省略则按模型自动） | 关 | — |
| HF transformers | `assistant_model=`（外挂），`prompt_lookup_num_tokens`（n-gram） | `num_assistant_tokens`、`num_assistant_tokens_schedule`、`assistant_confidence_threshold` | 关 | — |
| koboldcpp | `--draftmodel`（别名 `--model-draft` / `-md`） | `--draftamount`（别名 `--draft-max` / `--draft-n`） | 关 | — |
| TabbyAPI（exllama） | 配置段 `draft_model:`，`draft_mode: model / disabled / mtp / ngram`，`draft_model_name` | — | `model`（没填名字即关） | — |

## 发现
- **SPN-001 两种形状**：老一代是"指一个第二模型文件"（llama.cpp、LM Studio、kobold、mlx、HF：`draft model`）；新一代是"选一种方法"
  （vLLM `method`、SGLang `--speculative-algorithm`、TabbyAPI `draft_mode`、llama.cpp `--spec-type`），因为 2026 年起 MTP 内置在模型里，
  **没有第二个模型可指**。lmk 的 MVP 是 Qwen3.8 自带 MTP 头，属于后者。
- **SPN-002 词是 "draft"**：草稿模型、草稿 token，每家都用；"speculative" 只出现在区段标题或算法开关上。
- **SPN-003 每轮猜几个每家都暴露**：llama.cpp 默认 3、Ollama 4、mlx-lm 2、mlx-vlm 取草稿器 config（Qwen MTP 3）、SGLang 按模型自动。
  我们 exp02 实测：散文 2 最好，代码/复述 3 最好。
- **SPN-004 默认全部关**，例外是 Ollama 的 Gemma 4 MLX 路径（验证后改成默认开）和 TabbyAPI（模式默认 model，但没名字等于关）。
  "验证过的模型默认开"有先例（Ollama）。
- **SPN-005 统计**：LM Studio 报"接受的草稿 token 数 / 总 token 数"；mlx-vlm 有每请求的 rounds/accepted/drafted。
  lmk 的 usage chunk 已经带 `lmk.restore_ms`，加一个接受率是同一路数。
- **SPN-006 Ollama "投机解码属于模型不属于服务器"**，和 lmk "推荐按模型页"一致。
- **SPN-007 兼容判据的措辞**：LM Studio 对用户只说一句 "must have the same vocabulary as the main model"。我们的 MVP 不让用户配草稿模型，
  这句就不用出现在用户面。

## 对 lmk 的建议（进设计文档 §4 第 2 点）
- 配置项 `model.draft_model`，值 `auto` / `off`（MVP；以后再加 HF 仓或路径）。`draft_model` 是跨家族的词；`auto` = lmk pull 给这个模型带的草稿器
  （模型页说有没有）。不叫 `speculative_decoding: on`：用户搜的词是 draft model，且以后接外挂草稿时同一个键能装下。
- 高级项 `model.draft_tokens`（每轮猜几个），默认取模型页写的数（Qwen3.8-27B：3）；对应 mlx-vlm 的 `draft_block_size`。
- 默认 `off`；过验收的模型在模型页推荐 `auto`，seed 的 config 对默认模型写实际值。
- 状态行 `· draft on` + 自启动以来的接受率；usage chunk 加 `lmk.draft_accepted_tokens` / `lmk.draft_tokens`；bench 多打一行接受率。

## 裁定（09-23，owner）
建议里的 `draft_model: auto|off` **被否**：owner 问"why auto? instead of on?"——我把开关和将来的仓名塞进了一个键。且 YAML 1.1 把 `on`/`off`
解析成布尔，字符串键收不到 "on"。定为布尔 `speculative_decoding: true|false`（同 `thinking` 样式）+ `draft_tokens`；外挂草稿器将来另起
`draft_model:`。开在没有草稿器的模型上启动即报错退出（fail fast）。
