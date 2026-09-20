# exp01 — 脱离 LM Studio，直接驱动开源的 mlx-engine

日期 2026-09-19。lmk（lm-kitten，自建本地 LLM server）开 long run 之前的技术验证第一步。

## 目的
LMS-011 / LMS-014 的结论全是**读源码读出来的**。这里第一次真的跑：不经过 LM Studio 的 server，
用引擎的 Python 接口直接加载模型、生成、拿 prefill 进度和缓存命中数。回答四个问题：
1. 引擎能不能脱离 LM Studio 独立跑起来？
2. 磁盘前缀 cache 在独立进程里照常工作吗？分叉后的请求命中多少？
3. 命中 token 数、prefill 进度，能不能从引擎的回调里直接拿到（WISH-002 / WISH-005 的前提）？
4. prefill 速度和经 LM Studio 时持平吗（exp04：约 321 tok/s）？

## 方法
- 解释器与库：**借 LM Studio 自带的** cpython 3.11 + `app-mlx-generate…@34` 的 site-packages，
  `PYTHONPATH` 指过去，不安装任何东西。
- 模型：`Qwen3.8-27B-MLX-4bit`（16GB）。必须用带 `vision_config` 的模型——磁盘前缀 cache 只在
  `BatchedVisionModelKit` 那条路径里，纯文本模型走另一套。`max_kv_size=32768, max_seq_nums=4`。
  LM Studio 里同一个模型此刻也加载着（ctx 200000）——两份权重共 32GB，低于 77.8 GiB 上限。
- 三个请求，同一段 system（130 句合成填充文，估 2.5–3k token），`max_tokens=24, temp=0`：
  **A-cold**（问 alpha）→ **B-fork**（同 system，问 bravo：在 system 之后分叉）→ **A-again**（与 A 完全相同）。
- 自己实现一个 `PromptProgressReporter`，记下 begin 的 `cached_tokens` / `total_prompt_tokens`
  和每次 update。

## 预期（运行前写下）
1. 能跑起来；`load_model` 返回的是 `BatchedVisionModelKit`。把握中高——借来的环境就是 LM Studio
   生产上用的那套，但 LM Studio 的 worker 可能还做了我不知道的初始化。
2. **A-cold**：`cached=0`；update 每 2048 token 一次（本 prompt 约 3k ⇒ 1–2 次）；首 token 约 9–10 秒
   （3k ÷ 321）。
3. **B-fork**：`cached` = **2048**。公共前缀约 2.9k token，分叉点不在 checkpoint 上——A 的 checkpoint
   只在 2048 网格点和它自己的末块，末块含 A 的问题、B 用不上 ⇒ 退到 2048。把握中：这正是 LMS-005
   的规则，但我还没在"分叉"情形下亲眼验证过（exp05 的 6144 = 3×2048 是旁证）。
4. **A-again**：`cached` = floor256(prompt_tokens − 1)，即几乎整段命中；首 token < 2 秒。把握高。
5. 输出文本以 `<think>` 开头（引擎不剥思考）。把握高。
6. 和 LM Studio 里那一份同时驻留不出问题。把握中高。

## 对比基准
经 LM Studio 的 exp04（冷 prefill 321 tok/s）、exp03 / exp06（命中数与 256 块规则一致）。

## 运行
    ./run.sh
产物：`result.jsonl`（每步一行）、`stderr.log`。

## 结果（2026-09-19 18:0x，`result.jsonl` / `stderr.log` 原样保留）
| 步 | prompt | 回调里的 cached | prefill 进度 update | 首 token |
|---|---|---|---|---|
| A-cold | 2680 | 0 | 2048, 2560 | 8.38s（≈ 320 tok/s） |
| B-fork | 2680 | **2560** | 无 | 0.57s |
| A-again | 2680 | 2560 | 无 | 0.57s |

对照预期：
1. ✓ 独立跑起来了，`BatchedVisionModelKit`，加载 3.1 秒（权重已在系统页缓存里）。
2. ✓ A-cold：cached=0，320 tok/s——与经 LM Studio 的 321 **持平**。update 是 2048 和 2560：
   最后一步对齐到 ≤ prompt−1 的最大 256 倍数。
3. **✗ 我预期 B-fork 命中 2048，实际 2560。** 我以为 A 的末块 checkpoint 含着 A 的问题、B 用不上。
   错：末块 checkpoint 落在 floor256(prompt−1) = 2560，而两个 prompt 的差异（alpha / bravo）在 2560
   **之后**——那个边界之前全是共享的 system。所以分叉后的请求能恢复到
   "≤ 公共前缀、且存过 checkpoint 的最大边界"；只要前一个请求的非共享尾巴够短，这就是
   floor256(公共前缀)，损失 < 256 token，不是我在 WISH-019 / 流控设计里写的"至多 2047"。
   （回头看 exp05 的 6144：既等于 3×2048，也等于 floor256(约 6190)——当时我把它读成"退到 2048
   网格"是过度解读，两种解释它都符合。）
4. ✓ A-again：2560（= floor256(2679)），首 token 0.57 秒。
5. **✗ 输出不以 `<think>` 开头**，直接是思考内容（"We need answer user's request…"）。原因：这个
   模型的聊天模板把 `<think>\\n` 放在**生成提示**里（属于 prompt），模型是接着往下写；LM Studio 的
   兼容接口给补回了开标签。对 lmk：思考从第一个 token 就开始，只需找 `</think>`。
6. ✓ 与 LM Studio 里那一份同时驻留无异常。

## 读数
- **lmk 的地基成立**：开源引擎独立可用、速度持平、磁盘前缀 cache 照常工作。
- **WISH-002（usage 带命中数）与 WISH-005（带工具的流上有 prefill 进度）在引擎层面都是现成的**：
  `PromptProgressReporter.begin(cached_tokens, total_prompt_tokens)` + `update(prefill_tokens_processed)`。
- stderr 里引擎在抱怨 `Received a generation request without a request_id! Please send a request_id`
  ——**引擎本来就期待调用方给每个请求一个 id**，正好接 WISH-007 / kitten 的 refId。
- stderr：`VLM prompt cache disk budget: … cap_gib=162.81 limiter=max_kv_size`——磁盘 cache 的预算
  是按 max_kv_size 推出来的，可配。
- 预热（WISH-019）比我估的更划算：预热请求的假 user 消息只有几个 token，分叉点几乎总在末块
  边界之后 ⇒ 真实请求少命中 < 256 token（不到 1 秒），不是约 6 秒。
- N=1，2.7k token 的 prompt。没测：工具调用的原始文本与 `qwen3_coder` 解析器、并发请求、长 prompt。
