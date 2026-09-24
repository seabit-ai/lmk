# lmk backlog —— 现在到哪了、还欠什么

会变的东西放这里（`CLAUDE.md` 放不变的）。改了状态就回来改这份；条目做完就删，结论归到设计文档或 research。
最后核对：2026-09-24。

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
  散文 1.22× / 代码 1.59× / 复述 1.70×，灰区。新事实：mlx-vlm 已实现投机解码和 KV 量化，fork 只需接线。
- 09-23 SAD 已裁：KV 量化先；fork = `seabit-ai/mlx-engine` 的 `lmk` 分支按功能一 commit 随上游 rebase；验收四条（长上下文评测臂、探针实测 ctx、
  cache 身份带位宽、bench 只记录）；用户面 `model.kv_cache_bits` 16/8/4 默认 16、推荐按模型页、表显示推荐配置的数。
- 09-23 开工，KV 量化接线**已落**（`research/2026-09-23-kv-cache-quant/`）：fork 分支 `lmk` d3650db（引擎 +111 行、10 单测）；
  lmk 分支 `kv-cache-bits`（配置项、cache 身份、状态行、install/Makefile 指向 fork、`LMK_ITEST_KV_BITS`）。27B-4bit 8 位：itest 开关各 4/4 +
  重启还原；探针 34,816 B/token；bench 三数进表。**还差验收第 1 条**（长上下文评测臂，含 k8v4 变体）才能写进模型页推荐；
  之后是 Gemma 31B / 122B 的探针数与 README 表"推荐配置下的数"（`TestedModel.kv_cache_bits` 字段）。
  已知未优化：8 位下 cache 命中还原慢 19%（首 token 只差 0.02 s）；Qwen3.8 磁盘 cache 大头是 linear attention 检查点，量化只省 18%。
- 09-23 投机解码接线**已写、未上真机**：引擎 worktree `.engine/mlx-engine-spec` 分支 `lmk-spec`（b58a72e = `lmk` + 1，
  `batched_vision/speculative.py`，8 单测）；lmk 分支 `speculative-decoding`（e9a721d：`model.speculative_decoding` / `draft_tokens`、
  27B 四档指向 `seabit-ai/Qwen3.8-27B-MTP-draft`、`lmk pull` 顺带下草稿、无草稿即启动报错、状态与 usage 的接受数）。
  待办顺序：真机冒烟（`research/2026-09-23-speculative-decoding/exp03-engine-wiring/smoke.py`：贪心逐 token 一致、B=2、采样）→ itest
  `LMK_ITEST_DRAFT` → 引擎 `lmk` 分支 ff 到 lmk-spec、ENGINE_COMMIT 指过去 → owner 建 HF 组织后上传草稿器（模型卡与校验和已在
  `~/.cache/lmk-research/qwen3.8-27b-mtp-draft/`）→ 27B 模型页写推荐。本地测试用的草稿器已按 HF cache 目录结构摆在
  `models--seabit-ai--Qwen3.8-27B-MTP-draft/`（软链）。
  **真机结果（exp03）**：贪心下 code/copyedit 与普通解码逐字节一致，story 与 mlx-vlm 自己的循环同样分叉；采样 temp 1 接受率 87%。
  **验收全过（exp03 + exp04）**：单流草稿开着、默认采样：code 40/40、instruct 20/20、tools 10/10，接受率 86%，7 s/答对 9 s。27B-4bit 页已写推荐
  `speculative_decoding: true` 与 `kv_cache_bits: 8`（后者：exp02 长上下文 40/40、20/20、10/10）。引擎 `lmk` 分支已 ff 到 2839cfa，ENGINE_COMMIT 指过去。
  `speculative-decoding` 已合 main（2026-09-24，bda4425）。**09-24 发现并修了：两个开关同开会崩**（exp05，SPD-009）——fork `lmk` 分支 7a1e17f，
  `ENGINE_COMMIT` 已指过去（分支 `config-example`）。owner 机器已按推荐配置在跑（kv8 + 草稿 + thinking off）。
  **待 owner**：push 引擎 `lmk` 分支到 7a1e17f（先）再 push lmk；建 HF 组织 `seabit-ai` 后我上传草稿器
  （在此之前 `lmk pull` 会说"could not fetch it"，`speculative_decoding: true` 启动即报错说去 pull）；tag v0.7.0。
  **并发限制**：两条请求行长不齐时 mlx-vlm 的批量回滚不对（同 prompt 正确、code+story 第 53 个 token 分叉），MVP 只在单请求解码时投机
  （`speculative.py` 的 `MAX_ROUND_ROWS = 1`），多请求退回普通解码；复现脚本 `pair_probe`（scratch，内容见 exp03 README 第 5 条）。查清再开。
- 智能评测 `research/2026-09-23-intelligence-27b-vs-122b/`：三个 27B 臂已完（xhigh 最差、low 最好），122B-off 臂在跑；
  跑完要写四臂汇总 + "共同错的题"节，模型页据此更新（27B 的推荐配置可能改成 `reasoning_effort: low`）。

## 2026-09-24 `lmk up` 卡 10 分钟的事故——修了什么、还欠什么（分支 `speculative-decoding`）
- 现场：`~/.lmk/app/.engine/mlx-engine/` 是空目录但 `COMMIT` 标记说已装。04:48 的旧 installer `curl | tar` 下载 fork 上还没 push 的
  2839cfa，curl 404、tar 吃空流照样成功；05:19 的新 installer 因标记匹配整段跳过。服务 import 引擎失败 exit 1，launchd 每 30 s 拉起
  （152 次）；`lmk up` 只看"注册没注册"，等到 600 s 超时才报错。
- 修了（都有单测，真机照用户的样子跑过：崩溃循环与干净退出各 1–2 s 内报出，正常启动显示 "x of 15.0 GB"）：
  installer 标记匹配但 `generate.py` 缺也重装；`lmk serve` 引擎 import 失败 → `LmkRuntimeMissing`、exit 0；`lmk up` 读 launchctl 的
  pid / runs / last exit code（`service.job_state`），崩溃或退出立刻报，进度按服务进程常驻字节对模型权重字节（`memory.resident_bytes`，
  实测 rss 随权重读入从 0 到 16.5 GB / 9 s，权重 16.05 GB），不再显示秒数；`lmk status` 加载中也显示同一行，崩溃循环有专门的一行；
  报错里的 last events / last output 只取这次启动之后的行（旧进程的 LmkReady 和旧 traceback 不再混进来）。
- 候选（未裁，owner 问"还该修什么"时列的）：
  1. `lmk logs` 缺省只看 jsonl，traceback 只在 `--raw` 里——崩溃时用户要知道加 `--raw`（status 行已改指向）。要不要让 `lmk logs` 在
     最后一条事件之后把 stderr 的 traceback 一并打出来。
  2. 日志不轮转：`lmk.stderr.log` 已 420 KB、`lmk.jsonl` 185 KB，launchd 的 stdout/stderr 文件只会长。
  3. 体积单位：`lmk pull` 说 "16 GB"（十进制，HF 的数），加载进度和 status 的 memory 行说 "15.0 GB"（其实是 GiB，`human_bytes` 按 1024）。
     `config.yaml.example` 里已区分 "GB download / GiB loaded"，别处没有。
  4. `ENGINE_COMMIT` 指向 fork 上没 push 的 commit 时，`curl | sh` 装不上（installer 现在会明说）；从 checkout `make install` 不受影响。
     push 顺序（引擎先）已在上面"待 owner"里。
  5. `READY_TIMEOUT_S = 600` 现在只兜"加载真的卡住"这一种情况；卡住时用户看到的是进度数字不动。要不要改成"进度 N 秒没变就报"。
  6. 崩溃循环里 launchd 每 30 s 重拉一次直到有人管（`lmk status` 现在会说）。`lmk serve` 顶层再兜一层未知异常 → 记 `LmkCrashed` 后 exit 1
     保留重启，还是 exit 0 停下来，没裁。

## 2026-09-24 `config.yaml.example` 改成真配置（分支 `config-example`，现基于 main）
- 裁决与形状：OOBE 设计文档 C5；owner 09-24 三条纠偏已改（无分组行、模型只列名、`path:` 用 LM Studio 真实目录）。
  同一分支上带着 `ENGINE_COMMIT` → 7a1e17f（exp05 的修复，example 推荐的组合靠它才能跑）。待 owner 授权合并。
- **`lmk status` / `lmk bench` 跟上两个开关**（owner 09-24："I have no idea what's the current config"）：分支 `status-switches`（叠在 `config-example` 上，
  先合它）。status 加 `settings` 行三个开关永远写出；bench 首行报配置、行的模型格带开关、decode 分散文/代码并报接受率；三个日志事件补字段；
  README 的 status 示例换成真输出；27B 页 Speed 表加推荐配置一行（散文 44.5 / 代码 58.6，接受 88%）。已在 owner 机器上照用户的样子跑过。
- 种子 `config.yaml` 同形状（owner 09-24 "fix ~/.lmk/config.yaml too"）：注释文案与 example 共用一张表（`_WHAT`）；owner 的文件已照此重写，生效值不变。

## 2026-09-24 投机解码推到 122B（exp06，未成）与 Splash（research/2026-09-24-splash）
- 122B 的 MTP 头拆出来了（5.05 GB bf16，重排成 switch_mlp 布局才能加载；`~/.cache/lmk-research/qwen3.5-122b-a10b-mtp-draft`，HF cache 软链已摆）。
  冒烟：接受 0/2555，decode 0.52×，且 drafts 全拒时输出仍与普通解码分叉——两个嫌疑（草稿器权重映射 / MoE 校验前向），下一步用 mlx-vlm 自己的循环隔离。
  **没进 lmk**（models.py 未接 draft_repo），模型页不动。
- Splash：对手主张已记成 SPL-001..006，未验证。owner 定要不要同机实测（brew 装、约一小时、跑时要停常驻）。

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
