# lmk backlog —— 现在到哪了、还欠什么

会变的东西放这里（`CLAUDE.md` 放不变的）。改了状态就回来改这份；条目做完就删，结论归到设计文档或 research。
最后核对：2026-09-25。

## 现在到哪了
| 块 | 状态 | 出处 |
|---|---|---|
| 第一版 server（聊天 + 工具、思考分流、prefill 进度、调用方身份、warmup、持久化前缀 cache、图片输入） | 完成 | `docs/design/2026-09-19-lmk.md` |
| 开箱体验（`lmk pull/up/status/logs/down`、`~/.lmk/`、零必填配置的 example、HF 共享目录、`install.sh`） | 完成，已合 main | `docs/design/2026-09-20-lmk-oobe.md` |
| 内存护栏（加载前检查、准入队列四条规则、`requests.*` 三个配置、内存读数） | 完成，已合 main | `docs/design/2026-09-20-memory-guard.md` |
| 状态看板（`lmk status [-w]`：starting/prefill/decode、排队原因、刚结束的、cache 命中率） | 完成，已合 main | 同上 + `lmk/board.py` |
| 采样参数（temperature/top_p/top_k/min_p/repetition_penalty/stop；缺省读模型 `generation_config.json`；seed 收下不传、记 `LmkParamIgnored`） | 完成，已合 main | `docs/design/2026-09-21-sampling.md` |
| 预热降级为最低优先级 + decode/prefill 进度随流推送 + 预热耗时行 + floor256 修复（"warmed N new tokens in Xs" / "already warm · Xs" / "yielded after Xs"，只算真正落进检查点的量） | 完成，已合 main | `research/2026-09-24-prewarm/`；机制设计在 kitten `docs/design/2026-09-24-llm-progress.md` |
| 自己的引擎路线：KV cache 量化（16/8/4 位）、投机解码（MTP 头 + DFlash2 两种草稿器，`model.draft`）、mlx-vlm 升级到 0.6.16、校验改普通前向；27B-4bit 缺省草稿器 dflash2（其余位宽仍 mtp）。已知限制：投机只单请求一次一条（多行探过 exp15，最多 1.16×，不做，SPD-021） | 完成，已合 main | `docs/design/2026-09-23-own-engine.md`、`research/2026-09-23-kv-cache-quant/`、`research/2026-09-23-speculative-decoding/`、`research/2026-09-24-engine-upgrade-vlm616/` |
| 智能评测 27B vs 122B（122B 不更聪明、只更快更省；27B + `reasoning_effort: low` 最佳，xhigh 每类都更差且常超限） | 完成 | `research/2026-09-23-intelligence-27b-vs-122b/`，模型页已据此更新 |
| `lmk up` 自动下载缺失模型/草稿器 + Homebrew tap（`brew install seabit-ai/tap/lmk`，`seabit-ai/homebrew-tap` 已建） | 完成，已合 main，tap 已发布 | `docs/design/2026-09-20-lmk-oobe.md` §D2、`packaging/homebrew/README.md` |
| `lmk bench` 金丝雀校验 + `lmk report`（一块可贴的 Markdown：机器/构建/引擎/配置/status/金丝雀答案/最近事件） | 完成，已合 main | README Benchmarks 节 |
| kitten 调用方身份头（`X-Lmk-Purpose` / `X-Lmk-Ref-Id`），kitten 集成测试从 `/lmk/v1/status` 读回名字 | 完成，已合 kitten main | kitten `merge cce7615` |
| repo | `github.com/seabit-ai/lmk`，公开（2026-09-22 起，v0.1.0），MIT；当前 `v0.7.1`。`brew install seabit-ai/tap/lmk` 或 `curl \| sh`；owner 机器 `make install` 部署，launchd 常驻 | — |

测试：单测 214、跳过 8（`make test`，468ed98）。集成测试与真机验收的数字按功能分散记在各自 `research/` 里，不在此复述。



## 2026-09-25 长上下文与 KV 精度（`research/2026-09-25-spec-long-context`）
- 本机已改 kv16（SLC-007：128k 22.1 对 15.4 tok/s）。在做：MLX 缓冲池上限（128k 时进程 52–54 GB、MLX 峰值 42–43 GB，约 10 GB 是缓冲池；
  `mx.set_cache_limit` lmk 与引擎都没调）；补量 kv8 的进程内存，定"按内存自动选 KV 精度"的档位线（kv16 同时在内存的 token 1.84M → 0.98M）。
- 想法（别当计划）：融合反量化的 SDPA 向量 kernel——kv8 慢在算子拆分（每 1k 上下文每步 +0.30 ms 对 kv16 的 +0.15）；先看上游 MLX 有没有在做，不自己写。
- 想法：采样与贪心的差距在接受率（53% 对 59%，SLC-014），要追需 p/q 拒绝采样，另开题。

## 2026-09-25 带 tools 的请求用上投机解码（分支 `spec-with-tools`，fork `lmk-spec-proc` 42a248c）
- 已做：投机轮逐位过 logits processor（设计 `docs/design/2026-09-25-spec-with-tools.md`）；顺带修两个互相掩盖的老 bug：交接处 bonus token 喂两次（SPD-028）、
  在轮里结束的请求热 cache 比 all_tokens 少一个（SPD-032/033，真机 807 对 806）。真机 8/8 个 agent 步起草、接受 65%（SPD-035）。
- 跟进（未做）：
  - **SPD-034** kv8 下同一 prompt 从热 cache 续跑与从磁盘恢复续跑，首位置 logprob 差到 2.75（修复前后都有），可能也是"关草稿冷/热 sha 不同"的根。
    有鉴别力的实验：关草稿造热 cache 再与磁盘恢复比，kv16 各跑一次。
  - MTP 行逐位 lm_head 投影串行（可整块算）；DFlash `RoundResult.accepted` 现为 kept，与统计口径不同（无人读）。
- 结构化输出（分支 `structured-output`，设计已全关、计划 0d0c971）排在它之后，复用这套逐位 walk：计划里"约束期间关投机"一条要改成"照常起草"。

## 下一个：structured output（`response_format` / `json_schema`）
owner 已同意方向（2026-09-25），SAD 还没开题。引擎侧已有 `json_schema` 参数；与思考段、工具调用的关系没想清楚（口子留在 `docs/design/2026-09-21-sampling.md`）。

## 2026-09-24 `lmk up` 卡 10 分钟的事故——修了什么、还欠什么
修了（都有单测，真机验证过：崩溃循环与干净退出各 1–2 s 内报出，正常启动显示 "x of 15.0 GB"）：installer 标记匹配但引擎文件缺也重装；`lmk serve` 引擎 import 失败 → `LmkRuntimeMissing`、exit 0；`lmk up` 读 launchd 的 pid/runs/last exit code 立刻报崩溃或退出，不再等 600 s 超时；进度按服务进程常驻字节对模型权重字节算，不再显示秒数；崩溃循环有专门的状态行；报错里的 last events/output 只取本次启动之后的行。

候选（owner 2026-09-25 已裁一条，其余仍开）：
1. `lmk logs` 缺省只看 jsonl，崩溃时 traceback 只在 `--raw` 里——要不要在最后一条事件之后自动把 stderr 的 traceback 一并打出来。
2. 日志不轮转：owner 机器上约 5 天后 `lmk.jsonl` 382 KB、`lmk.stderr.log` 717 KB，仍在长大。
3. 体积单位：`lmk pull` 说的 "GB"（十进制，HF 的数）与加载进度/status 的 "GB"（其实是 GiB，`human_bytes` 按 1024）不是一回事；`config.yaml.example` 里已区分 "GB download / GiB loaded"，别处没有。
4. 崩溃循环里 launchd 每 30 s 重拉一次直到有人管（`lmk status` 会说）；`lmk serve` 要不要在顶层再兜一层未知异常记 `LmkCrashed` 后 exit 1 保留重启，还是 exit 0 停下——没裁。

**否决（owner，2026-09-25）**：候选 5「进度 N 秒没变就报」——原则是把进度实时报给用户，让用户自己判断是不是错误，lmk 不该替用户猜原因。`lmk up` 已经在实时显示加载进度，够了。

## 122B 投机解码：试了，没接
`research/2026-09-24-splash/`：122B 的 MTP 头拆出来能加载，但接受 0/2555、校验输出与普通解码分叉，两个嫌疑没查清（草稿器权重映射 / MoE 校验前向）。没进 `models.py`，模型页不动。

## 等 owner 定的
- owner 机器上 `~/.lmstudio/models/` 里那份同款模型（15G）已无人使用，删不删。
- Splash（对手）的主张记成 SPL-001..006，未验证；要不要同机实测（brew 装、约一小时、跑时要停常驻）。

## 没验证过的（写进 README 的都验证过；这些没有）
- **重启机器后 launchd 自动拉起**——自第一版起就没验证过。下次重启后 `lmk status` 看一眼。
- 小内存 Mac：准入规则三（token 账）只有单测；加载前检查的"装不下"分支只有单测。我们只有一台 96GB 的机器。
- 准入规则四（内存压力危急）在真机上从未触发——要触发就得把机器压到危急，没做。
- cache 取回的代价只量到 27k token（约 0.37 秒；约 75ms + 10ms/1k）；更长的上下文、**重启机器后的真冷读**没量。
- 并行：27k 会话只量了 2 个同时；4 个没量。短 prompt 量了 1/2/4。
- 已实测家族：Qwen3.8（27B，4/5/6/8 位）、Qwen3.5-122B-A10B（4bit/48GB）、Gemma 4（12B/26B-a4b/31B/e4b；12B 是 tried-not-listed）。纯 attention（无 hybrid/线性注意力）模型上与 oMLX 的差距是否消失——预期消失，仍未专门量过。
- 等待中的客户端断开：断开检测目前只在流式生成阶段起作用（写失败即 `ClientGone`）；还在准入队列里排队的请求感知不到断开，会占着队位直到被放行后第一次写失败、或等满 `max_wait_seconds`。没加探测。
- PDF 输入不支持。

## 想法（没裁过，别当计划）
- 给上游 mlx-engine 提"持久化前缀 cache"：评论稿在 `research/2026-09-20-mlx-engine-upstream/`（针对其 issue #354），是否已由 owner 发出未确认。
- 给 oMLX 提 issue："混合架构模型在每次请求末尾补一个 256 边界的 snapshot"——它在这类模型上每步慢约 9 秒的原因（SVY-006）。
- `lmk status` 的 `starting` 阶段要不要细分出 `restore`（`restoreMs` 已在日志里，看板上还是笼统写 starting）。
- 本地 cache 不过期、命中代价约为重算的 0.5%（MG-009）——对 agent 侧"为保 cache 不敢动历史"的设计是个不同的前提（kitten 的 cache-aware compaction）。
