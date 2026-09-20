# exp01 — oMLX 对 lmk：同一台机器、同一个 27B、kitten 的真实请求

日期 2026-09-20。回答 SVY-003 留下的问题：oMLX 在**我们这个场景**（Qwen3.8-27B 混合架构、56 个工具、
约 11k 的固定前缀、服务重启后的第一句话）到底行不行。

## 方法
- 机器 m3u（M3 Ultra 96GB）。模型 `~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit`（两边同一份权重）。
- **oMLX**：上游 `14194fe`（0.7.0.dev4，2026-09-18），`pip install -e` 进 scratchpad 的 venv；
  `omlx serve --base-path <scratch>/base --model-dir <scratch>/models --port 8011 --paged-ssd-cache-dir <scratch>/cache`。
  全新空 cache。不碰机器上已有的 oMLX 0.3.2（`~/.omlx`，端口 8000）。
- **lmk**：常驻服务（端口 1235，main 上的版本），cache 已有内容。
- 载荷取自 kitten 的 wire dump `.kitten/logs/1789871561139/000016-request.json`（发给 lmk 的真实请求）。
  **载荷和原始输出含私人内容（邮件摘要），不在本 repo**，在 `~/src/nova/2026/2026.0920.M3uOmlxVsLmk/`。
  生成方式：`jq '.body | .messages = .messages[0:2] | .stream=true | .stream_options={include_usage:true}'`（首轮），
  长会话版去掉 `.messages` 截断：
  - `payload-first-turn.json`：前 2 条消息（system + "hi"）+ 56 个工具 —— 新会话的第一句。
  - `payload-long.json`：完整 54 条消息 —— 一段带工具往返的真实会话。
- 脚本 `run.py`：流式请求，记 TTFT（第一个带内容的 chunk：reasoning / text / tool_call 任一）、总时长、usage。
- 步骤（GPU 上串行，一次只跑一个请求）：
  - A1 首轮，空 cache（冷）
  - A2 同一请求再发一次
  - A3 分叉：同前缀，user 消息换成一句要求读文件的话（同时考工具调用）
  - A4 长会话整段
  - **重启 oMLX 进程**
  - B1 首轮请求（重启后第一句）
  - B2 长会话整段
  - lmk 跑 A2/A3/A4 同款作对照；lmk 的冷启动与重启后数字用已有记录（LMK-005：冷 36.8s；重启后 11008/11174，TTFT 2.4s）。

## 预期（跑之前写的）
1. oMLX 0.7 能加载 Qwen3.8-27B，走 VLM 引擎。把握 70%（四月的 0.3.2 肯定不行，所以才装新版）。
2. A1 冷 TTFT 30–45s（两边底下是同一套 MLX kernel，lmk 冷是 36.8s）。
3. A2 命中 ≥ 90% 的 prompt，TTFT < 3s。把握 85%。
4. A3 分叉命中 ≥ 90%（前缀 11k 里只有最后一条 user 不同）。把握 65%——混合架构要靠 boundary snapshot，
   #3699 说的就是这类模型上 snapshot 没存下来。
5. **B1 重启后命中 ≥ 90%：把握 40%**。这是 oMLX 的主打卖点，但 SVY-003 的三个未关 bug 都落在
   "混合架构 × SSD cache" 上。不中的话，预计表现是 cached=0 或只命中到某个很小的边界。
6. A4 / B2 长会话：若 #3608（命中上限约 25k）仍在，长会话的命中会被截在 25k 左右。载荷 token 数待测，
   若不足 25k 则此项测不到。
7. 工具调用：返回结构化 `tool_calls`、参数是合法 JSON。把握 80%。reasoning 单独放在 `reasoning_content`。把握 70%。
8. usage 里有 `prompt_tokens_details.cached_tokens`。把握 75%（源码 `openai_models.py:421` 有这个字段）。

## 结果
原始输出：nova `2026/2026.0920.M3uOmlxVsLmk/raw/`（每次调用的 SSE + `results.jsonl`）。oMLX 日志在 scratchpad，
关键行已抄进下文。

| 步骤 | oMLX 0.7.0.dev4 命中 / prompt | oMLX TTFT | lmk 命中 / prompt | lmk TTFT |
|---|---|---|---|---|
| A1 首轮，空 cache | 0 / 11212 | 39.98s（含加载模型） | —（LMK-005：冷 36.8s） | — |
| A2 同一请求再发 | 8192 / 11212（73%） | 9.85s | 10240 / 11139 | 4.21s |
| A3 分叉 + 工具调用 | 8192 / 11201 | 9.81s | 11008 / 11128（99%） | 0.81s |
| A4 长会话首发 | 8192 / 27263 | 64.13s | 27136 / 27190（已在 cache 里，不可比） | 1.53s |
| A5 长会话再发 | 24576 / 27263（90%） | 9.98s | 27136 / 27190（99.8%） | 0.89s |
| B1 **重启后**首轮 | 8192 / 11212 | 12.86s（含加载模型约 3s） | —（LMK-005：11008/11174，2.4s） | — |
| B2 **重启后**长会话 | 24576 / 27263 | 9.88s | — | — |

decode 速度两边一样：oMLX A5 374 tok / 11.4s ≈ 33 tok/s；lmk A4 349 tok / 10.6s ≈ 33 tok/s。
同一份消息 oMLX 数出的 prompt 比 lmk 多 73 token（模板渲染有差别，没追）。
oMLX 测试 cache 占盘 2.4G（约 33k token 的内容，6 个 4096 块 + snapshot）。

**读法：** 用户关心的是出字要等多久。稳态每次调用 oMLX 约 10s、lmk 约 1s；重启后第一句 12.9s 对 2.4s。在这个场景里 lmk 是更好的方案，差一个数量级。

### 对照预期
1. ✅ 能加载，走 VLM 引擎（`type: vlm, engine: vlm`），并打了 `Qwen3.5/3.6 GDN prefill kernel patch`。
2. ✅ 冷 40s。
3. ❌ **A2 只命中 73%，TTFT 9.85s**，不是 ≥90% / <3s。原因见下。
4. ❌ A3 同样 8192（73%）。不是 #3699 那种"存不下来"，是粒度。
5. ❌ **猜错，错在悲观一侧**：重启后命中与重启前完全一样（8192、24576）。oMLX 的跨重启持久化在这个混合架构
   模型上是真的能用。我给的 40% 低估了它。
6. ⚪ 测不到：prompt 27k，floor4096 = 24576，刚好在 #3608 说的 25k 之下。
7. ✅ 结构化 `tool_calls`、参数合法 JSON；reasoning 单独走 `reasoning_content`。
8. ✅ `usage.prompt_tokens_details.cached_tokens` 有，且与它自己的日志一致。

### 差距的来源：块粒度，是设计取舍不是 bug
oMLX 启动日志原话：`Enlarging paged cache block_size=256 to 4096 for ArraysCache hybrid model (reduces boundary
snapshot overhead)`；存的时候 `storing 8192/11301 tokens (skipping trailing partial block, 1 intermediate snapshots)`。
源码 `omlx/scheduler.py:2878`：`target = max(_ARRAYS_CACHE_BLOCK_SIZE(=2048), prefill_step_size, _qwen35_prefill_floor)`，
只增不减，**没有配置项能调小**。所以对 Qwen3.5/3.8 这类模型，命中只能落在 4096 的整数倍上：
每次调用要重算 `prompt mod 4096` 个 token，平均约 2048（≈6.4s），最坏 4095（≈12.8s）；本次两个载荷都是约 3000（≈9.8s）。
mlx-engine / lmk 的做法是 KV 按 256 分块、recurrent state 只在 2048 网格 **加上每次 prompt 的 floor256(末尾)** 打 checkpoint，
所以"同一会话往后追加"这条最常走的路径损失 <256 token（≈0.8s）。
