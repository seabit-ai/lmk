# 再支持哪些模型

日期 2026-09-22。起因：owner "what other LLM we can add support?"。先摆门槛与本机候选，再挑。

- **MDL-001 三道门槛。** ① 磁盘前缀 cache 只在引擎的 VLM 路径（`BatchedVisionModelKit`，config 带 `vision_config`；LMK-001）——
  纯文本模型进来就没有 lmk 的核心卖点；② 引擎在这条路径上"验证过"的家族只有 `qwen3_5`/`qwen3_6` 与 `gemma4`
  （`batched_vision/context_fit.py:224-227`），别的家族窗口拟合走通用回退，能不能跑要试；③ 工具调用解析靠 mlx-lm 从
  chat template 推断（`_infer_tool_parser`），思考段切分靠 lmk 的 splitter 认标记——换家族两处都要验。
- **MDL-002 本机已有的候选（不用下载）。** `~/.lmstudio/models/`：Qwen3.8-27B-MLX-8bit（28G，同家族，零代码）；
  Qwen3.5-122B-A10B-4bit（65G，`Qwen3_5MoeForConditionalGeneration`，256 专家，有 vision）；baa-ai 的
  Qwen3.5-122B-A10B-RAM-48GB（44G，同架构混合量化）。HF cache 里 gemma-4-31b-it / 26b-a4b 与 Qwen3.5-122B 只有空壳目录。
  纯文本的 Llama-3.1-8B、Qwen2.5-7B 过不了门槛 ①。
- **MDL-003 只在一台 96GB 机器上测过**是 README 已承认的缺口；给 32/64GB Mac 的小模型是受众最大的空白，
  但 Qwen3.5/3.6/3.8 家族有哪些小尺寸、是否带 vision，未查（别凭印象写型号）。
- **MDL-004 Qwen3.5-122B-A10B 不做（owner 提醒；我漏了）。** 2026-09-19 四臂评测（kitten `research/2026-09-19-lmstudio-provider` LMS-007）：
  8 个任务合计 27B-4bit 103s / 3.9k token、指令遵循 3/3；122B 标准版 356s / 20.6k token、2/3；122B-48GB 430s / 22.4k、2/3。
  122B 每个小任务先想 1–4k token，端到端慢 3–4 倍，指令遵循更差；owner 当天卸掉换回 27B-4bit。
  "MoE 激活少所以快"在任务级延迟上被思考量反转。条件：temp 0、思考全开、单次。

## 裁决（owner，2026-09-22）
先做 Qwen3.8-27B-MLX-8bit（同家族，本机已有）。
