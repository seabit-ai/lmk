# exp02：Qwen3.5-122B-A10B-MLX-4bit 能不能进 lmk 的实测表

日期 2026-09-22。m3u（M3 Ultra 96GB，GPU 工作集上限 77.76 GiB），权重 `~/.lmstudio/models/mlx-community/Qwen3.5-122B-A10B-4bit`（65 GB）。
常驻的 27B-4bit 服务测试期间停掉（65 + 15 > 77.8）。lmk 分支 `model-122b`。

## 预期（跑之前写）
1. **加载**：引擎的窗口拟合会把 262k 降下来——65 GB 权重后剩约 10 GiB 给 KV 与中间量。把握：高。降到多少不猜。
2. **集成测试 5/5**（工具调用、cache 命中、warmup、图片、跨重启）：同家族（qwen3_5_moe，引擎验证过），模板与解析器同。把握：中高——
   MoE 走 state_checkpoint 那套 cache 没试过。
3. **decode**：每 token 读约 5 GB（激活 10B @4bit）对 27B 的 16 GB ⇒ 理论 3 倍，路由与 256 专家的开销打折 ⇒ **60–80 tok/s**（27B 为 39）。把握：中。
4. **冷 prefill**：算力 ∝ 激活参数（10B 对 27B）⇒ 快 2 倍以上，**500–800 tok/s**。把握：中低——MoE prefill 的专家分发效率不明。
5. **思考长度**：MDL-004 里"每个小任务想 1–4k token"是 LM Studio 下、贪心。这里缺省按模型 generation_config 采样；bench 的 decode
   探针（250 字故事）看 reasoning 长度，与 27B 同题对比。把握：低，只记录。

## 结果（2026-09-22，raw/）
| | 预期 | 实得 |
|---|---|---|
| 加载 | 窗口被降 | 33 s；262,144 → **165,888**（baseline 64.8 GiB，KV 24 KB/token，比 27B 的 64 KB 小） |
| 集成测试 | 5/5 | **5/5**，160 s |
| decode | 60–80 | **60.5 tok/s** |
| 冷 prefill | 500–800 | **753 tok/s** |
| cached prefill | — | 89k tok/s（KV 文件比 27B 小 2.7 倍，取回更快） |

思考长度（temp 0，max_tokens 6000，raw/thinking-lengths.txt、raw/122b-*.json）：
| 题 | 27B 思考开 | 122B 思考开 | 122B 思考关（`model.thinking: false`） |
|---|---|---|---|
| 250 词故事 | 3,524 token，答 1,341 字符 | **6,000 撞上限，无答案**——思考里把故事写了几稿后逐词点数 "The(1) wind(2)…" | 402 token，315 词，完整 |
| 三条要点 | 123 token | 645 token | 63 token，正好三条 |

1–4 全部命中。5：MDL-004 的"想太多"复现了，触发点是题目里的数量约束（250 词），与 THK-003（手点 128 项清单）同一模式；
Qwen3.5 的模板只有 `enable_thinking` 开关、没有力度档位（27B 的 Qwen3.8 模板才有 `reasoning_effort`）。关掉思考后 122B 又快又听话。
思考关时工具调用没测（集成测试是思考开的）。

### 追加（owner 问："题目说 250 词"——该不该说"大约 250 词"？）
同模型、思考开、temp 0、max_tokens 6000（raw/122b-story-about.json、raw/122b-story-noc.json）：
| 题 | 思考 | 结果 |
|---|---|---|
| "Write a 250-word story…" | 逐词点数 | 6000 撞上限 |
| "Write a short story of **about** 250 words…" | **仍逐词点数** | 6000 撞上限 |
| "Write a short story about a lighthouse keeper."（无字数） | 3,416 字符，不点数 | 1,844 token，704 词，完成 |
"大约"救不了：只要题目里有数量，它就去核对（思考里自己写的就是 "Approximately 250 words"，照数不误）。去掉数量才不数。
⇒ 对这个模型，数量约束 + 思考开 = 高风险组合；bench 的 decode 探针（250 词故事，max_tokens 400）只量速度不看内容，不受影响。
