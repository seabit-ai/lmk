# 同类方案调研：谁能做"不限时的、落在磁盘上的前缀 cache"

日期 2026-09-20。起因：owner 想把 lmk 打磨后开源，判断是"lmk 是我所知唯一能做不限时磁盘前缀 cache 的方案"；
我前一天提的第一条顾虑正是"这句话我没查过"。编号前缀 `SVY-`。**全部是读网页得到的二手信息，没有一条上手验证过。**

## 发现

### SVY-001 "唯一"不成立：至少三个公开项目自称前缀 cache 落盘且跨重启
| 项目 | 许可 / 规模 | 跨重启的磁盘前缀 cache | 出处 |
|---|---|---|---|
| **oMLX**（jundot/omlx） | Apache 2.0；GitHub 约 21.9k star；issue 编号已到 #3770 | README 原话："restored from disk instead of recomputed from scratch - **even after a server restart**"；vLLM 式分块、内容哈希、RAM 热层 + SSD 冷层（safetensors）；`--paged-ssd-cache-dir` | https://github.com/jundot/omlx |
| **vMLX**（jjang-ai/vmlx） | MIT | 官网原话："Spills to SSD in fixed-size pages at full precision … so a long conversation **survives a restart**" | https://vmlx.net/ |
| **llama.cpp** `--slot-save-path` + **stillwarm** | MIT；stillwarm 是 2026-07 的小包装 | llama-server 的 `/slots/<id>?action=save|restore`，整段序列状态落盘、跨重启；stillwarm 把它自动化（按模型 SHA + 前缀哈希校验），自报 71–201× 提速。单 slot、手动式，不是内容寻址的分块 cache | https://huggingface.co/blog/vimalnakrani/stillwarm-kv-cache-persistence |
| LM Studio（mlx-engine） | 引擎 MIT，应用闭源 | **明确不跨重启**。官方博客（2026-06-05）原话："the cache is temporary and will not leave persistent files"；卸载时 "clears its in-memory index and closes the scratch file"。未提任何持久化计划 | https://lmstudio.ai/blog/mlx-engine-agentic-workloads |

### SVY-002 oMLX 的功能面远大于 lmk
（均出自其 README）OpenAI **与 Anthropic** 两套 API；工具调用覆盖 mlx-lm 的全部格式（点名 Qwen3.5 的
`<function=…>`）；VLM 与多图；embeddings / rerank；多模型同驻 + LRU 驱逐 + 钉住 + 每模型空闲 TTL；
**总内存上限**（缺省"系统内存 − 8GB"，防整机 OOM——lmk 没有的护栏）；连续批处理；投机解码；macOS 菜单栏
应用 + `/admin` 网页；dmg / Homebrew / pip 三种装法。

### SVY-003 混合架构（Qwen3.5 这类带 recurrent state 的）是各家的共同难点，oMLX 也有未关的 bug
oMLX 有"boundary snapshots"机制（相当于 mlx-engine 的 state checkpoint），但 issue 列表里（2026-09 抓取）：
#3699 "SSD prefix cache never stores on qwen4_exp hybrid: available_boundaries=0 while boundary snapshots are
enabled"（open）；#3608 "Prefix-cache match capped at ~25k tokens on verified linear (append-only)
conversations"（open）；#3690 "TurboQuant KV cache (4-bit) silently corrupts detail retrieval for requests that hit
the prefix cache"（open）。llama.cpp 一侧：对 SWA / 混合模型会打印 "forcing full prompt re-processing due to lack
of cache data (likely due to SWA or hybrid/recurrent memory)"，Qwen3.5 需要 `--swa-full`；混合模型的状态
save/restore 曾经崩溃、近期才确认可用。
**没查到的**：oMLX 在 Qwen3.8-27B 上、带工具、长前缀、重启后的真实命中率——这只能上手测。

### SVY-004 我们自己四月的 omlx 实验记录没找到
research-discipline 技能里举的例子是 `research/2026-04-04-omlx-cache/`，Bruce 09-19 也说"其它的 framework
还是不行"；但 `~/src/workspace/research/` 和各仓库的 research/ 下都没有这个目录。五个月前的结论也可能已过时
（oMLX 的 issue 号显示它迭代极快）。

### SVY-005 上手实测：oMLX 在我们的场景里**能用**，跨重启持久化是真的（exp01）
oMLX 0.7.0.dev4（上游 `14194fe`）加载 Qwen3.8-27B-MLX-4bit 正常；56 个工具的真实 kitten 请求，工具调用解析正确、
reasoning 分离、usage 报 cached_tokens；服务重启后命中与重启前一致（首轮 8192/11212，27k 长会话 24576/27263）。
SVY-003 担心的三个 bug 本次一个都没撞上（#3608 的 25k 上限因载荷长度刚好测不到）。
**我事前给"重启后命中 ≥90%"的把握是 40%，低估了它。**

### SVY-006 差距在粒度：oMLX 对混合架构把 cache 块放大到 4096 token，每次调用多等约 9 秒
同一份载荷，稳态（cache 已热）下的 TTFT：oMLX 9.8–10.0s，lmk 0.8–1.5s。原因不是 bug：oMLX 对带 recurrent
state 的模型把块从 256 放大到 4096（写死的下限，无配置项，`omlx/scheduler.py:2878`），尾部不满一块的不存；
lmk 靠 mlx-engine 在每次 prompt 的 floor256(末尾) 打 state checkpoint，追加式会话损失 <256 token。
decode 速度两边相同（≈33 tok/s）。对 agent 循环的含义：一个 10 次工具往返的 turn，oMLX 比 lmk 多等约 90 秒。
对纯 attention 模型（oMLX 块 = 256）这个差距应当不存在——**未测**。

### SVY-007 m3u 上早就装着 oMLX 0.3.2，SVY-004 找不到的"四月实验"现场就是它
`~/.omlx`（2026-04-04 建）：launchd `com.seabit.omlx` 常驻 8000 端口，`~/models` 下 gemma-4 ×2 + Qwen2.5-7B，
累计 624 次请求（几乎全是 gemma-4-26b，cached/prompt = 96%），SSD cache 占 176G。实验笔记本身仍没找到。
机器侧的处置（一个重复的 launchd 任务循环了五个月，已停）记在 nova `2026/2026.0920.M3uOmlxVsLmk/notes.md`，不在本 repo。

## 这对 lmk 意味着什么（判断，不是事实）
- "唯一能做不限时磁盘前缀 cache"——**不成立**。这是 Apple Silicon 本地推理圈 2026 年的热点，oMLX 把它做成了
  主打卖点，规模和成熟度都远在 lmk 之上。
- lmk 可能仍有的差异点（**都未经对比验证**）：建在 mlx-engine 上，混合架构的 checkpoint 逻辑是 LM Studio
  的工程师写的、我们实测过命中规则（LMK-002）；usage 里的命中数与流上的 prefill 进度；单模型常驻的极简；
  一千五百行、看得完。
- 开源一个"和 21.9k star 的项目主打同一个卖点、但功能少一个数量级"的东西，价值有限；更有价值的两条路：
  ① 上游那条评论（UPS-003）——把持久化送进 LM Studio 的引擎，受益面最大；② 先弄清 oMLX 在我们这个
  场景下到底行不行——行，就该认真考虑直接用它；不行，"不行在哪"才是 lmk 真正的立身之本。

- **上手之后的判定（SVY-005/006）——以用户的位置看，不以"功劳归谁"看**：
  owner 要的是"在 m3u 上跑 qw38-27B 的最好体验"。实测每次调用 lmk 0.8–1.5s 出字、oMLX 约 10s；重启后第一句 2.4s 对 12.9s。
  差一个数量级，且压在 agent 循环的每一次调用上。**所以在这个场景里 lmk 就是最好的方案**——"唯一能跨重启"不成立，
  "体验最好"成立。这个优势底下是 mlx-engine 的 checkpoint 策略而非 lmk 自己写的代码，对用户毫无区别；
  我上一版把它写成"可证实的优势只剩一条、还不是 lmk 自己的"，是拿工程师的记功账本替用户下结论（纠偏见 corrections log）。
- 同一顶帽子下，oMLX 有而 lmk 缺、且用户**感受得到**的：内存护栏（09-19 整机卡死那类事故的保险）、采样参数
  （temperature 等现在被忽略）。用户感受不到的（多模型、Anthropic API、菜单栏应用）不急。
- 开源价值要重估：卖点不是"磁盘 cache 能跨重启"（oMLX 也能），而是"混合架构模型上 agent 循环每步亚秒"——这是可以
  用本实验复现的、对 Qwen3.5/3.8 用户实打实的差异。

## 未做
- ~~上手对比~~ → exp01，SVY-005..007。
- exp07 同款的完整 agent 任务（多轮工具往返的端到端耗时）没跑，只测了单次调用；"多等约 90 秒"是按单次差值外推的。
- oMLX 在 >25k（#3608）、并发、图片输入下的表现；纯 attention 模型上两边的差距。
- vMLX 的混合架构与工具调用支持（官网没写）。
- mlx-lm 自带 server 现在的缓存能力（只知道它有内存 LRU prompt cache）。
