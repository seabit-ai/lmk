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
- **MDL-005 27B-8bit 实测（exp01）。** 集成测试 5/5（工具调用、cache 命中、warmup、图片、跨重启；温度 0）。同条件对照 4bit：
  decode 22.9 对 39.3 tok/s，冷 prefill 318 对 325 tok/s，权重 29.5 GB 对 16.1 GB。
  顺带发现：采样合并后集成测试变随机（缺省 temp 1.0），8bit 第一次把图里的 4217 读成 4917；集成测试改为 temperature 0 后两模型全过。
  采样合并当时没有重跑 `make itest`（CLAUDE.md 规则 5 的漏洞），这次补上：4bit 5/5。

## bench（owner 裁：固定模板 + 随机数打头使 cache 失效；`lmk bench` + `docs/benchmarks.md`）
- **BNC-001 空闲九小时后的第一次触碰慢三倍。** 常驻 4bit 服务空闲约 9 小时、其间另外加载过三份权重（itest、exp01）之后，
  bench 的冷 prefill 第一次 110 tok/s（4,074 token 首 token 37.0 s；引擎的 prefill 计划与 restoreMs 都正常），紧接着第二次 319 tok/s。
  解释：权重被系统换页/压缩，第一次触碰读回来——与 LM Studio 时代"切换慢是权重 page-in"一致。没直接量页面状态，是从两次差推的。
  ⇒ bench 先发一个小预热请求并单独报它的首 token 时间，正式数在预热之后量。
- **BNC-002 bench 的两行（同机、同日）**：4bit 323 tok/s / 1.02 s / 39.5 tok/s；8bit 319 / 1.07 / 23.1。与 exp01 服务端日志的数一致（318/325、22.9/39.3）。
- 事故：给临时 8bit 服务收尾用 `pkill -f "lmk serve"`，把 launchd 的常驻服务一起杀了（干净退出，按设计不自动重拉）；当时无人在用，
  `lmk up` 拉回，看板计数清零。教训进 CLAUDE.md：临时服务记 PID 按 PID 杀。
- **MDL-006 MiniMax-H3 不是语言模型。** HF 标签 text-to-video / image-to-video（diffusers），lmk 跑不了；owner 问起，记一笔。
  MiniMax 的语言模型是 M 系列（M2.1/M2.5/M3，openclaw 里走云端）；有无 MLX 权重、是否带 vision_config 未查。

## 候选二轮（2026-09-22，HF 实查）
- **MDL-007 Qwen3.8 只有 27B 一个尺寸**（官方仓库：27B、27B-FP8、Flash-Next、2.4T-A95B；MLX 版只有 27B 的 4/5/6/8bit）。
  给小内存 Mac 的选择要往 Qwen3.6 或 Gemma 4 找。
- **MDL-008 三个过门槛的候选**（都带 vision_config，都是引擎验证过的家族）：
  | 模型 | 家族 | 权重 | 窗口 | 备注 |
  |---|---|---|---|---|
  | lmstudio-community/gemma-4-12B-it-MLX-4bit | gemma4_unified | 6.7 GB | 262k | 16/32GB Mac 的候选；家族不同，工具调用与思考标记要重做 |
  | lmstudio-community/gemma-4-E4B-it-MLX-4bit | gemma4 | 6.8 GB | 131k | 同上，更小 |
  | lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit | qwen3_5_moe | 20.4 GB | 262k | 同家族零解析工作；MoE 激活 3B ⇒ decode 应快于 27B；质量与"想太多"（MDL-004 的 122B 教训）要测 |
  未列：Qwen3.6-27B（和 3.8-27B 同尺寸同家族，只是旧一代，没有增量）。
- **MDL-009 122B-A10B 实测（exp02）**：lmk 零代码改动即可跑（引擎认 qwen3_5_moe）。集成测试 5/5；96 GB 机器上窗口降到 165k；
  decode 60.5 / prefill 753 / cached 89k tok/s。"想太多"复现且定位到触发点（数量约束 → 逐词点数），Qwen3.5 模板只有思考开关；
  关思考后 250 词故事 402 token 完成。⇒ 进实测表，备注写明窗口与思考开关；顺手落地了裁而未做的 `model.thinking` / `model.reasoning_effort`。
  MDL-004 的判定修正为：122B **思考全开**时不适合 agent 小任务；思考关或（若模板支持）低档时未在 agent 任务上评测。
- **MDL-010 122B 关思考后工具调用正确（exp03）**：集成测试 5/5；两步读改文件、不该用工具时不用、整数参数类型，全对，每步 26–57 token。
  ⇒ 122B 在 agent 上的可用姿势是 `model.thinking: false`；真实长任务上的质量仍未评。
- **MDL-011 122B 的 48GB 混合量化版（exp04）**：路由专家 2–3 bit、注意力/共享专家 5–8 bit；44 GB，96 GB 上窗口保住 262k。
  集成测试开关思考各 5/5；三个 agent 任务 3/3；decode 53.7（比标准版慢 11%，与"字节少就快"的预期相反）、prefill 746、cached 89k。
  它回答的是"122B 能否上 64 GB Mac"——从内存看能（43.9 GiB baseline + KV），但没有 64 GB 机器实测；作者是个人（baa-ai），下载 484。
  ⇒ 进表，备注写明 2 bit 专家、出处、速度不升反降；"Not tested" 写 64 GB 机器与难任务质量。
- **MDL-012 Gemma 4 26B-A4B 进 lmk（exp05）**：第二个家族。Dialect（思考标记/缺省/每轮必思考/消息形状）落地；集成测试开关各 5/5；
  agent 任务 3/3；decode 113.5、冷 prefill 约 1,850、cached 87k；15 GB 权重、262k 窗口。三条 Qwen 隐含假设被翻出（F1–F3）。
  Gemma 的 `thinking: false` 是提示不是硬开关（F3）。
- **MDL-013 小 Gemma（exp06）**：E4B 进表（16 GB 组；agent 任务四轮全对；decode 93.5 / prefill 2,199）；12B 不进表（工具结果后贪心循环、
  漏参数；F6）。切分器改为正文阶段也认思考开标记（F5）。详情在 `docs/models/gemma-4-12b-4bit.md`（owner：不进 README，别污染客户的阅读上下文）。
- **MDL-014 模型表改版（owner 意见）**：去掉仓库/GiB 等工程字段；按"至少 N GB"分组，列出该机型上的上下文（引擎公式 + 实测系数，96 GB 用实测）、
  图片、思考、三个速度、一句"适合什么"。每个模型页加 "Known issues" 节（owner 提议）。
- **MDL-015 Gemma 4 31B（exp07）**：owner 纠偏"模型不是按尺寸计价的商品"后实测。集成测试开关各 5/5；agent 任务 9/9；decode 33 / prefill 252 /
  cached 26k（KV 80 KB/token，列表最大）；32 GB 组。进表：同尺寸档的第二个家族选择，速度与 cache 不如 27B，质量未比。
