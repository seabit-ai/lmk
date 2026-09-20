# 自建本地 LLM server 的愿望清单

日期 2026-09-19 起，**持续追加**。owner："I think we already collected lots of items in our wish
list. You should take notes on all of them. Those can influence our design."
背景：lmstudio provider 第一版已合 main；"基于开源 mlx-engine 自建 server、替掉 LM Studio"挂在
`docs/design/2026-09-19-lmstudio-provider.md` 的"门"里（可行性见 LMS-011：磁盘 KV cache、批处理、
工具协议都在 MIT 的开源引擎里）。这份清单是那场设计开题时的输入。编号前缀 `WISH-`，只增不改号。

每条写三样：**疼在哪**（今天的实证，指到出处）、**想要什么**、**备注**（难点 / 未核实的）。
排序不代表优先级；优先级等开题时再裁。

## A. 前缀 cache

### WISH-001 前缀 cache 跨加载、跨重启持久化
- 疼在哪：LM Studio 的 cache 寿命 = 模型加载期。底层是 `tempfile.TemporaryFile`（无名临时文件，
  进程退出即消失，`prompt_cache/blob_store.py:84`），索引和 LRU 只在内存。日志自己标着
  `lifetime=model_load`；09-19 每次加载后都从 `records=2` 重新攒起。重启 m3u = 全部清空，
  kitten 每个 session 的第一轮重新冷算约 11k token 的固定前缀（exp07：首 token 55 秒）。
- 想要什么：cache 存成有名字的持久文件 + 落盘的索引；重启后前缀照样命中。
- 备注：key 已经是纯内容哈希（LMS-005），持久化时 key 里还得带上模型身份、量化、引擎/格式版本，
  否则换了权重会命中错的 KV。要有容量上限和淘汰（现有 LRU + 字节预算可沿用）。

### WISH-002 usage 里带缓存命中数
- 疼在哪：两套接口的 usage 都没有 cached token 数，只在 server log 的 DEBUG 行里（LMS-008/010）。
  kitten 只能把它标成"未上报"（`Usage.CachedTokensUnreported`），cache-aware compaction 的证伪
  日志对本地模型失明。
- 想要什么：每次响应的 usage 带 `cached_tokens`。
- 备注：引擎的 prefill begin 事件里本来就有 `cached_tokens` / `total_prompt_tokens`（LMS-011），
  是 LM Studio 的接口层丢掉的。

### WISH-003 调用方可声明"分叉点"，让 checkpoint 落在那里
- 疼在哪：混合架构（Qwen3.5：3/4 的层是线性注意力）只能在存过 state checkpoint 的边界恢复；
  checkpoint 只落在 2048 token 的预填充网格 + 每个 prompt 的末块（LMS-005/010）。分叉点不在
  这些边界上，就多重算至多 2048 token。
- 想要什么：请求里能标"这里以后会分叉"（典型：system prompt + 工具定义的末尾），server 在那里
  存一个 checkpoint。
- 备注：相当于 Anthropic `cache_control` 断点的本地版。收益要实测——2048 token 约 6 秒。

### WISH-004 不止一份热缓存
- 疼在哪：内存里的热缓存"exactly one"（`coordinator.py`），新请求一来就让位；多 session 交替
  时全部落到磁盘路径（256 块粒度）。
- 想要什么：按内存预算保留 N 份热缓存。
- 备注：磁盘路径实测也就 2–3 秒，这条的收益可能不大，开题时先量。

### WISH-019 "touch"：只预热前缀、不生成
- 疼在哪：重新加载 / 重启之后（WISH-001 解决之前），以及每个新 session，kitten 约 11k token 的
  固定前缀要冷算 40–55 秒（exp07），这段时间用户在干等第一句回答。
- 想要什么（owner 2026-09-19 提出）：一个只做 prefill、不生成的调用——把给定前缀算完、落进
  cache，并**在前缀的末尾存一个 checkpoint**，让之后任何以它开头的请求整段命中。
- 备注：**今天用 LM Studio 就能近似做到**——普通聊天请求 + `max_tokens:1`。预热请求得带一条假的
  user 消息，真实请求在那里分叉；restore 落到"≤ 公共前缀、且存过 checkpoint 的最大 256 边界"。
  假消息只有几个 token 时，那就是 floor256(公共前缀)——**少命中 < 256 token，不到 1 秒**
  （lmk 验证 exp01 / LMK-002；我最初写的"至多 2047 token、约 6 秒"是把规则读错了）。
  自建 server 的 touch 配合 WISH-003（在指定位置存 checkpoint）可以一个 token 都不丢。
  kitten 侧：它是第四种"用途"，且可让路——真正的 turn 一到就取消，已落盘的块照样被用上。
  只对本地有意义：云端 cache 有 TTL、预热要钱。

## B. 可见性

### WISH-005 带工具的那条流上要有 prefill 进度
- 疼在哪：进度事件只在 LM Studio 自己的接口上，而那套接口不收自定义工具（LMS-010）。agent 必须
  用兼容接口 ⇒ 几十秒的冷 prefill 和引擎卡死，从外面看一模一样（lmstudio 设计 §7.3 明说的缺口）。
- 想要什么：同一条 SSE 流里既能带工具，也发 prefill 的 start / progress / end。
- 备注：引擎有现成的回调（每 2048 token 一次）。

### WISH-006 思考在 server 端分流
- 疼在哪：兼容接口把思考混在 `content` 里的 `<think>…</think>`，kitten 得自己写流式分流器
  （`internal/provider/lmstudio/think.go`）；LM Studio 自己的接口倒是分好了。
- 想要什么：带工具的接口也把思考作为独立的 delta 发出来。
- 备注：引擎的 `tool_protocols.py` 已经知道各模型族的 reasoning 标记。

### WISH-007 调用方身份：谁、为了什么、关联 id
- 疼在哪：exp07 里三次调用打到同一个模型（groom、turn 的两次），从 server 一侧看一模一样；
  我是靠 kitten 自己的 wire dump 才认出抢 GPU 的是 groom。换成 Bruce 的调用就无从查起。
- 想要什么：每个请求带**谁**（客户端、项目/实例）、**为了什么**（turn / groom / compaction /
  测试…）、**关联 id**（W3C `traceparent`——kitten 每次调用本来就有 traceId，journal 里也有）。
  server 逐请求打进日志，和 kitten 的日志、journal 对得上。
- 备注：owner 提出。kitten 侧对应 `ChatRequest` 上的"用途"字段（流控 SAD 的 A 点）。

### WISH-008 "此刻谁在用 GPU"与队列状态可查
- 疼在哪：只能靠 `lms ps` 的 STATUS 一列猜；共享引擎上 Bruce 和 kitten 互相看不见。
- 想要什么：一个状态接口：加载着什么、窗口多大、正在为谁算什么、排队多深。
- 备注：kitten 可以据此在活动卡上显示"模型正忙，排第 N 个"。

### WISH-009 日志：结构化、持久、stderr 不进黑洞
- 疼在哪：openclaw 的 gateway stderr 指向 `/dev/null`，锁 watchdog 的释放行永远看不到；它的
  结构化日志在 `/tmp`，一重启就没了（08:13 那次无响应因此查不出原因，BHC-006）；LM Studio 的
  命中行要开 DEBUG 才有。
- 想要什么：每请求一行结构化日志（调用方、命中/未命中 token、各阶段耗时、是否被取消），
  落在重启不丢的地方，stderr 也落盘。

## C. 调度

### WISH-010 按用途排队：前台优先，后台让路
- 疼在哪：exp07，groom 和前台 turn 抢同一块 GPU，冷 prefill 从约 321 掉到约 203 tok/s。
  进程内的流控只管得了一个 kittend；每个项目各一个 kittend，再加 Bruce，互相看不见。
- 想要什么：server 看得见所有调用方，按用途调度——前台 turn 优先；后台的活空闲时才开始，
  前台一来就让路（取消或暂停）。
- 备注：依赖 WISH-007。只排队没用：后台先到时，前台排队等它和并发抢 GPU 差不多久（流控设计 §0）。

### WISH-011 取消要真的释放 GPU
- 疼在哪：客户端断开后 LM Studio 是否真的停止生成，未核实（lmstudio 设计 §3.4）。
- 想要什么：连接断开 / 显式取消后，计算立刻停，GPU 立刻让出来。
- 备注：WISH-010 的"让路"靠它。

## D. 模型加载与容量

### WISH-012 声明式的常驻模型：启动即按指定窗口加载
- 疼在哪：LM Studio 开机只起 server，不加载模型；JIT 加载用多大窗口文档没写（LMS-002），JIT 的
  模型 60 分钟 TTL 后被卸掉、cache 随之清空；kitten 按设计永不触发加载。现在靠我写的一个
  LaunchAgent 脚本补这个洞（`~/.kitten/bin/lmstudio-ensure-model.sh`）。
- 想要什么：配置里写"常驻这个模型、窗口这么大"，server 启动即加载；没有 JIT，没有 TTL 驱逐。

### WISH-013 护栏对着真实上限算，且不能被一个 json 文件关掉
- 疼在哪：m3u 的 GPU 工作集上限是 77.8 GiB，不是 96（LMS-006）。Bruce 看到"59GB 空闲页"就断言
  装得下，改 `settings.json` 把护栏从 high 关到 off 去加载 65 GiB 的模型，事后也没改回（BHC-006）。
- 想要什么：加载前对着 `max_recommended_working_set_size` 和已驻留的模型算账；要绕过就得人来、
  显式地做。
- 备注："权限是提醒"——这条防的不是恶意，是一个判断错了的 agent。

### WISH-014 内存估算要把窗口算进去
- 疼在哪：`lms load --estimate-only` 对 32k / 64k / 131k / 262k 四档窗口返回的全是 20.97 GiB，
  还标着 Confidence: LOW。
- 想要什么：估算 = 权重 + 按窗口算的 KV cache + 激活；给出"这个窗口下还能同时驻留什么"。

### WISH-015 窗口的语义要诚实：要么是硬上限并报错，要么如实说不是
- 疼在哪：按 32768 加载的模型，36970 token 的 prompt 完整算完、完整可读、无任何提示
  （exp05/06，LMS-012）。kitten 的 reactive 压缩靠一个 typed 的 too-long 错误触发，在这里永远等不到。
- 想要什么：明确的语义 + 超限时一个 typed 的错误（能映射成 `LlmContextTooLong`）。
- 备注：超窗很多时的行为没测。

## E. 请求级的控制

### WISH-016 按请求开关思考、限制思考预算
- 疼在哪：Bruce 的评测里端到端耗时由思考 token 数主导——122B 每个小任务先想 1–4k token，
  有一条想到 5999 token 撞上限、没有答案（LMS-007）；exp07 里 groom 整理一个空 session 也
  输出了 736 token。经兼容接口能不能关思考，未核实。
- 想要什么：请求参数控制思考的开 / 关 / 上限。groom、摘要这类活大概率该关。

### WISH-017 保住已经好用的：OpenAI 形状的工具调用
- 现状（不是疼点，是要保住的）：并行 tool call、按 index 分片、`finish_reason:"tool_calls"`、
  回传闭环，27B-4bit 经兼容接口全部可用（exp02/03/07）。
- 想要什么：自建 server 在这条线上与 OpenAI 形状保持兼容——kitten 的 lmstudio provider 几乎
  不用改就能接上。

### WISH-018 图片 / PDF 走同一条带工具的接口
- 现状：第一版丢弃附件并 WARN（lmstudio 设计 §4 明确后置）。三个 Qwen 都是 VLM。
- 想要什么：附件可用时不必换接口。优先级低。

## F. 顺带记下的、属于 kitten 自己的后续（不是 server 的事）

这些不进自建 server 的设计，但同一天发现，记在这里防丢；各自的出处是权威记录。
- groom 整理了一个空的、刚建的 session，且 daemon 一启动就跑——LMS-013。
- LLM 调用的流控（进程内）——`docs/design/2026-09-19-llm-call-flow-control.md`，SAD 进行中。
- kitten 自己的搜索工具（付费 API）——lmstudio 设计的"门"，等本地模型用一段再开题。
- cache-aware compaction——设计点全关、未开工。
- webview 的 `usagefold.ts` 孪生没跟上"命中数未上报"，其"缓存被打破"探测器会对本地模型误报。
- 有工具把二进制文件（约 1MB 的 PDF / PNG）当文本读进了 transcript——CCE-005 旁见，未查。
- kitten 没有 stall 看门狗：引擎卡死时 turn 挂到用户打断为止（云端同样）。
