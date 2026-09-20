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

## 未测
- 工具调用的刁钻情形：嵌套对象参数、值里含标记字样、格式写坏时语法约束能否兜住、别的模型族
- 同一进程内的并发请求（`max_seq_nums`）、长 prompt（>32k）、图片输入
- 机器重启后 launchd 自启 lmk 的完整链路；客户端断开后引擎是否立刻让出 GPU（WISH-011）
