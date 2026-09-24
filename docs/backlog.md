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
| repo | `github.com/seabit-ai/lmk`，**公开**（2026-09-22，v0.1.0），MIT（Copyright Seabit AI）。owner 的机器上由 `make install` 部署、launchd 常驻 | — |

测试：单测 122（无 GPU）、集成测试 5 + cache 兼容 1（真实模型）。

## 下一个：（空）——采样参数已做完，在分支 `sampling` 上待合并
- 做了：`temperature`/`top_p`/`top_k`/`min_p`/`repetition_penalty`/`stop`；缺省读模型的 `generation_config.json`；
  stop 由 lmk 在回答段匹配；seed 收下不传并记日志。取证 `research/2026-09-21-sampling`（SMP-001..004，exp01/02）。
- 没做（单独一题）：`response_format` / `json_schema`（引擎有 `json_schema` 参数）。

## 引擎路线（SAD 开题 2026-09-23，未裁）——`docs/design/2026-09-23-own-engine.md`
- owner 要投机解码 + 磁盘 cache 同时有、KV 量化。我方案：fork mlx-engine 在批处理路径补，不从零写。
- 09-23 量了（`research/2026-09-23-speculative-decoding/exp02`）：mlx-lm 路径对 Qwen3.8 不可用；用自带 MTP 头经 mlx-vlm 量得
  散文 1.22× / 代码 1.59× / 复述 1.70×，灰区。新事实：mlx-vlm 已实现投机解码和 KV 量化，fork 只需接线。我倾向 KV 量化先，待裁。
- 智能评测 `research/2026-09-23-intelligence-27b-vs-122b/`：三个 27B 臂已完（xhigh 最差、low 最好），122B-off 臂在跑；
  跑完要写四臂汇总 + "共同错的题"节，模型页据此更新（27B 的推荐配置可能改成 `reasoning_effort: low`）。

## 已裁但还没做的
- **kitten 发 `X-Lmk-Purpose` / `X-Lmk-Ref-Id`**：2026-09-20 已在 kitten repo 的分支 `llm-call-identity` 上实现并验证
  （用途 `turn` / `compaction` / `groom`；refId = `<sessionId>/<actionRef>`，groom = `groom/<project>/<startMs>`；未声明的不发头）。
  kitten 的集成测试从本服务的 `/lmk/v1/status` 读回了名字。待 owner 合并、装新 kitten 并重启 kittend 后，他的会话在看板上才有名字。

## 等 owner 定的
- owner 机器上 `~/.lmstudio/models/` 里那份同款模型（15G）已无人使用，删不删。

## 没验证过的（写进 README 的都验证过；这些没有）
- **重启机器后 launchd 自动拉起**——自第一版起就没验证过。下次重启后 `lmk status` 看一眼。
- 小内存 Mac：准入规则三（token 账）只有单测；加载前检查的"装不下"分支只有单测。我们只有一台 96GB 的机器。
- 准入规则四（内存压力危急）在真机上从未触发——要触发就得把机器压到危急，没做。
- cache 取回的代价只量到 27k token（约 0.37 秒；约 75ms + 10ms/1k）；更长的上下文、**重启机器后的真冷读**没量。
- 并行：27k 会话只量了 2 个同时；4 个没量。短 prompt 量了 1/2/4。
- 其它模型家族（只跑过 Qwen3.8-27B-4bit）；纯 attention 模型上与 oMLX 的差距是否消失（预期消失，未量）。
- 等待中的客户端断开：感知不到，会占着队位直到被放行后第一次写失败或等满 `max_wait_seconds`。没加探测。
- PDF 输入不支持。

## 公开（2026-09-22 已公开，v0.1.0）
- 真名清理、agent 片段、`curl | sh` 实测都已完成。`curl | sh` 从公开仓库装到临时目录 25 秒，装到 v0.1.0；
  发现并修了一个 heredoc 反引号 bug（分支 `install-heredoc`）。CI 首跑通过（1m17s）。
- 未验证之一"`curl | sh` 这条安装路径"由此关闭；其余未验证项不变。

## 想法（没裁过，别当计划）
- 给上游 mlx-engine 提"持久化前缀 cache"：评论稿在 `research/2026-09-20-mlx-engine-upstream/`（针对其 issue #354），是否已由 owner 发出未确认。
- 给 oMLX 提 issue："混合架构模型在每次请求末尾补一个 256 边界的 snapshot"——它在这类模型上每步慢约 9 秒的原因（SVY-006）。
- 投机解码（引擎有 `load_draft_model`）；`lmk status` 的 `starting` 阶段要不要细分出 `restore`（`restoreMs` 已在日志里）。
- 本地 cache 不过期、命中代价约为重算的 0.5%（MG-009）——对 agent 侧"为保 cache 不敢动历史"的设计是个不同的前提（kitten 的 cache-aware compaction）。
