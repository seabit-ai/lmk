# lmk backlog —— 现在到哪了、还欠什么

会变的东西放这里（`CLAUDE.md` 放不变的）。改了状态就回来改这份；条目做完就删，结论归到设计文档或 research。
最后核对：2026-09-20。

## 现在到哪了
| 块 | 状态 | 出处 |
|---|---|---|
| 第一版 server（聊天 + 工具、思考分流、prefill 进度、调用方身份、warmup、持久化前缀 cache、图片输入） | 完成 | `docs/design/2026-09-19-lmk.md` |
| 开箱体验（`lmk pull/up/status/logs/down`、`~/.lmk/`、零必填配置 + 自动刷新的 example、HF 共享目录、`install.sh`） | 完成，已合 main | `docs/design/2026-09-20-lmk-oobe.md` |
| 内存护栏（加载前检查、准入队列四条规则、`requests.*` 三个配置、内存读数） | 完成，已合 main | `docs/design/2026-09-20-memory-guard.md` |
| 状态看板（`lmk status [-w]`：starting/prefill/decode、排队原因、刚结束的、cache 命中率） | 完成，已合 main | 同上 + `lmk/board.py` |
| repo | `github.com/seabit-ai/lmk`，私有，MIT（Copyright Seabit AI）。owner 的机器上由 `make install` 部署、launchd 常驻 | — |

测试：单测 122（无 GPU）、集成测试 5 + cache 兼容 1（真实模型）。

## 下一个：采样参数（已排队，未开题）
- 现状：请求里的 `temperature` / `top_p` / `seed` / `stop` 等**被忽略**（README 的 "What does not work yet" 第一条）。
- 已就位：`Engine.generate(..., sampling: dict)` 直通引擎（兼容测试在用 `{"temp": 0.0}`）。缺的是 HTTP 层解析与校验。
- 引擎 `create_generator` 接受：`temp`、`top_p`、`top_k`、`min_p`、`min_tokens_to_keep`、`seed`、`repetition_penalty`、
  `repetition_context_size`、`stop_strings`、`top_logprobs`、`json_schema`（结构化输出）、`max_tokens`。
- 要先过设计的点：OpenAI 名字 → 引擎名字的映射（`temperature`→`temp`、`stop`→`stop_strings`）；缺省值（模型自带的 generation_config？）；
  `seed` 是"对该模型此后所有生成"生效的（引擎文档原话），并发下的含义；`stop` 与三路切分器的关系；`response_format` / `json_schema` 要不要一起。

## 已裁但还没做的
- **思考力度是 server 级常量**（裁于 kitten repo 的 `docs/design/2026-09-19-llm-call-flow-control.md` §6.2）：配置里加
  `model.reasoning_effort` / `model.enable_thinking`，启动时读一次、对所有请求一致。依据：模型模板把它渲染在 prompt 第 0 块，
  中途换档 = 整段对话冷算（`research/2026-09-19-lmk-spike` LMK-008）。原裁决写的配置位置是 `~/.kitten/lmk.yaml`，现应落在 `~/.lmk/config.yaml`。
- **kitten 发 `X-Lmk-Purpose` / `X-Lmk-Ref-Id`**：2026-09-20 已在 kitten repo 的分支 `llm-call-identity` 上实现并验证
  （用途 `turn` / `compaction` / `groom`；refId = `<sessionId>/<actionRef>`，groom = `groom/<project>/<startMs>`；未声明的不发头）。
  kitten 的集成测试从本服务的 `/lmk/v1/status` 读回了名字。待 owner 合并、装新 kitten 并重启 kittend 后，他的会话在看板上才有名字。

## 等 owner 定的
- owner 机器上 `~/.lmstudio/models/` 里那份同款模型（15G）已无人使用，删不删。

## 没验证过的（写进 README 的都验证过；这些没有）
- **重启机器后 launchd 自动拉起**——自第一版起就没验证过。下次重启后 `lmk status` 看一眼。
- `curl … | sh` 这条安装路径（repo 私有，测不了）；从 clone 里 `./install.sh` 已验证（21 秒）。
- 小内存 Mac：准入规则三（token 账）只有单测；加载前检查的"装不下"分支只有单测。我们只有一台 96GB 的机器。
- 准入规则四（内存压力危急）在真机上从未触发——要触发就得把机器压到危急，没做。
- cache 取回的代价只量到 27k token（约 0.37 秒；约 75ms + 10ms/1k）；更长的上下文、**重启机器后的真冷读**没量。
- 并行：27k 会话只量了 2 个同时；4 个没量。短 prompt 量了 1/2/4。
- 其它模型家族（只跑过 Qwen3.8-27B-4bit）；纯 attention 模型上与 oMLX 的差距是否消失（预期消失，未量）。
- 等待中的客户端断开：感知不到，会占着队位直到被放行后第一次写失败或等满 `max_wait_seconds`。没加探测。
- PDF 输入不支持。

## 公开之前要做的
1. 旧文档里的真名：`docs/design/2026-09-19-lmk.md`、`2026-09-20-lmk-oobe.md` 及早期 research 共约 40 行，还有 `/Users/<name>/…` 路径。
   两条路：逐处换成 "owner"；或公开版不带 `docs/design` 与 `research`。**已 push 的历史里也有**——要干净就得重建历史后强推
   （repo 刚建、只有 owner 在用，代价小）。owner 拍板。
2. owner 的 GitHub 用户名也算真名：现存一处，在 `research/2026-09-20-mlx-engine-upstream/notes.md`（grep 他的用户名即可找到）。
3. README 链到 `research/2026-09-20-local-server-survey/`（中文笔记）；若 research 不随公开版走，链接要改。
4. 公开后实测 `curl | sh`。

## 想法（没裁过，别当计划）
- 给上游 mlx-engine 提"持久化前缀 cache"：评论稿在 `research/2026-09-20-mlx-engine-upstream/`（针对其 issue #354），是否已由 owner 发出未确认。
- 给 oMLX 提 issue："混合架构模型在每次请求末尾补一个 256 边界的 snapshot"——它在这类模型上每步慢约 9 秒的原因（SVY-006）。
- 投机解码（引擎有 `load_draft_model`）；`lmk status` 的 `starting` 阶段要不要细分出 `restore`（`restoreMs` 已在日志里）。
- 本地 cache 不过期、命中代价约为重算的 0.5%（MG-009）——对 agent 侧"为保 cache 不敢动历史"的设计是个不同的前提（kitten 的 cache-aware compaction）。
