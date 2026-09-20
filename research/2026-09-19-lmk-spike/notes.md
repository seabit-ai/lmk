# lmk（lm-kitten）技术验证

日期 2026-09-19。lmk = 基于开源 mlx-engine 自建的本地 LLM server，替掉 LM Studio 的闭源那一层。
开 long run 之前的验证。编号前缀 `LMK-`。上游：research/2026-09-19-lmstudio-provider（LMS-011 / 014）、
research/2026-09-19-local-llm-server-wishlist（WISH-）。

## 发现

### LMK-001 开源 mlx-engine 可脱离 LM Studio 独立驱动，速度持平，磁盘前缀 cache 照常工作（exp01）
借 LM Studio 自带的 Python 与库、不装任何东西，`load_model` + `create_generator` 直接跑 27B-4bit：
冷 prefill 320 tok/s（经 LM Studio 是 321）；同前缀的第二个请求首 token 0.57 秒；命中 token 数和
prefill 进度从 `PromptProgressReporter` 回调里直接拿到；引擎期待每个请求带 `request_id`。
磁盘前缀 cache **只在 VLM 那条路径**（`BatchedVisionModelKit`，模型 config 带 `vision_config`）——
纯文本模型走另一套、只有内存 cache。

### LMK-002 分叉后的命中规则（更正 LMS-005 的读法）
请求的 state checkpoint 存在 2048 网格点 **和 floor256(prompt−1)**。后来的请求恢复到
"≤ 公共前缀、且存过 checkpoint 的最大 256 边界"。前一个请求的非共享尾巴短于末块余量时，那就是
floor256(公共前缀)，损失 < 256 token；否则才退到 2048 网格。exp01 的 B-fork：预期 2048，实得 2560。

### LMK-003 这个模型的思考开标签在 prompt 里，不在输出里
聊天模板的生成提示以 `<think>\\n` 结尾；引擎输出从思考内容直接开始，只有 `</think>` 出现在输出里。
LM Studio 兼容接口里看到的 `<think>` 是它补的。

### LMK-004 工具调用这一块是现成的：模型自带模板渲染 + mlx-lm 的 `qwen3_coder` 解析（exp02）
27B 写出的原文是 `<tool_call><function=…><parameter=key>⏎value⏎</parameter>…`，两个调用并行；
`mlx_lm.tool_parsers.qwen3_coder.parse_tool_call(block, tools)` 两块都解析成功，且按工具 schema 把
整数 / 布尔 / JSON 数组字面量还原成了正确类型，返回形状就是 OpenAI `tool_calls[].function`。
lmk 自己只需写：流式三路切分（思考 / 正文 / 工具调用块）+ 调用 id。引擎接受 `request_id`。

### LMK-005 干净环境 = 同一个引擎：公开仓库 + pip，数字与借来的环境逐项一致（exp03）
mlx-engine main @ `08f0c07`，上游 `requirements.txt` 全钉版本（`mlx==0.32.0`，mlx-lm / mlx-vlm 钉 commit），
python 3.11 venv 一次装成（1.2G，torch 占大头）。同一份 spike.py：prompt 2680 token、命中 0 / 2560 / 2560、
首 token 8.65s / 0.57s / 0.59s——与 exp01 一致（冷 prefill 差 3%，单次测量）。
⇒ lmk 可以完全不依赖 LM Studio 的安装；依赖的钉法可直接沿用上游。

### LMK-006 lmk 第一版端到端：重启之后第一句话从盘上命中，首 token 55s → 2.4s
lmk 以 launchd 服务常驻（`~/.kitten/lmk/app`，27B-4bit，窗口 200000，端口 1235）；kitten 经新的 `lmk`
provider 连它，跑 exp07 同款任务（读三个文件、指出真 bug）。同一段 kitten 固定前缀（约 11.2k token）：

| 情形 | 第一次调用命中 | 首 token | 整个 turn |
|---|---|---|---|
| exp07：经 LM Studio，启动时 groom 抢 GPU | 0 | 55.2s | 约 70s |
| lmk，cache 为空 | 0 / 11175 | 36.8s（≈304 tok/s） | 约 60s |
| **lmk，服务重启之后（全新进程、全新 kitten session）** | **11008 / 11174** | **2.4s** | 约 32s |

重启时日志：`LmkStopping`（排空 cache I/O 线程）→ `LmkCacheRestored records=80 mib=3635`。
kitten 的用量行显示真实的 `in 11.8k (cache 93%) · ctx 6%`——命中数来自 usage，不再是"未上报"。
答案正确（点名 7 号茶壶的华氏度，依据是 data model 的摄氏度约定），三个 file_read 并行发出。
口径：N=1；"重启"是 `launchctl kickstart -k` 重启服务，不是重启机器（机器重启后的自启链路未验证）；
第二行里 kitten 启动时的 groom 仍会打到 lmk 上（LMS-013 未修），这次没有与主调用重叠。
我中间一次重跑没测到东西：CLI 重新 attach 旧 session 回放了旧用量行，脚本误判完成——清掉 session 后重跑才是上表第三行。

### LMK-007 kitten 固定前缀的布局：工具定义在最前、占 96% 的稳定段；会变的记忆段只占尾部 4%
渲染后的 prompt（Qwen3.8 的聊天模板，exp02 的 `rendered-prompt.txt`）顺序是：
`<|im_start|>system` → 模板自己的一句"思考力度"开场白（LMK-008）→ `# Tools <tools>…</tools>` →
**然后才是** kitten 的 system 文本 → 对话。kitten 的 system 文本内部顺序（cmd/kittend/systemprompt.go:82）：
框架 + 能力 + scopes → AGENTS.md → **记忆段**（USER.md、facts INDEX、今天与昨天的 daily）→ skills 菜单。
量了一次真实请求（kitten 项目，`.kitten/logs/…/000009-request.json`）：工具定义 JSON 37,156 字符；
system 文本 7,487 字符，其中记忆段起点在 5,537——**工具 + 记忆段之前的 system 合计占固定前缀的 96%**，
记忆段及其后（1,950 字符，约 500 token）占 4%。
含义：前缀 cache 是 256 token 一块的哈希链（LMS-005），记忆一变，只有从变化点往后的块失效。持久化
之后（LMK-006），新 session 的第一次调用即使记忆刚变过，要重算的也只是尾部约 500 token（1–3 秒），
不是 11k（37 秒）。会让大头失效的只有：工具集变了（换 kitten 版本）、AGENTS.md 改了、cache 被淘汰、
模板开场白变了（LMK-008）。旁见：工具定义段不含项目路径 ⇒ **不同项目（kitten / nova）共享这 80% 以上的前缀**。
口径：字符不是 token；只量了一个请求。

### LMK-008 这个模型的聊天模板自带"思考力度"开关，而且它写在 prompt 的最开头
`chat_template.jinja:46-54`：`enable_thinking`（缺省开）与 `reasoning_effort`（`xhigh` 缺省 / `medium` /
`low`）；选中的力度渲染成 system 块的第一句话（"Reasoning effort is set to xhigh. Please think carefully…"）。
两个含义：① WISH-016（按请求控制思考）在模板层面是现成的——lmk 只要把它作为模板变量传进去；今天所有
调用都在缺省的 **xhigh** 下跑（Bruce 评测里"每个小任务先想几千 token"与此吻合，未单独验证因果）。
② **它在 prompt 的第 0 个块里**：换一档力度 = 整条前缀哈希链换 key，11k token 全部冷算。kitten 的
`ChatRequest.Effort` 注释说"请求参数，不进 prompt，切换不破 cache"——对云端成立，对这个本地模板不成立。
要用它，得当成 session 级的固定值，不能每次调用随意切。

## 未测
- 工具调用的刁钻情形：嵌套对象参数、值里含标记字样、格式写坏时语法约束能否兜住、别的模型族
- 同一进程内的并发请求（`max_seq_nums`）、长 prompt（>32k）、图片输入
- 机器重启后 launchd 自启 lmk 的完整链路；客户端断开后引擎是否立刻让出 GPU（WISH-011）
