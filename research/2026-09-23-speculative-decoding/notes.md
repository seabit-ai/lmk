# 投机解码（speculative decoding）能不能给 lmk 用

日期 2026-09-23。起因：owner "你听说过投机解码吗？…what are the candidate draft models for qw38-27B?"

- **SPD-001 当前引擎（08f0c07）在 lmk 走的路径上不支持。** `batched_vision/model_kit.py:388-397`：`BatchedVisionModelKit.load_draft_model`
  直接 raise "Speculative decoding is not currently supported for batched vision models"，`is_draft_model_compatible` 恒 False。
  实现只在顺序路径（`model_kit.py`，纯文本、无磁盘 cache）。⇒ 在这个引擎版本上，**投机解码与磁盘前缀 cache 二选一**，lmk 拿不到。
- **SPD-002 兼容判据 = 词表大小相同**（`model_kit.py:200-201`：`draft_tokenizer.vocab_size != self.tokenizer.vocab_size` ⇒ 不兼容）。
  Qwen3.8-27B 词表 248,320；同词表的小模型：Qwen3.5-0.8B / 2B / 4B / 9B（都是 qwen3_5、带 vision）、Qwen3.8-Flash-Next（qwen4_exp）。
  MLX 4bit 版在 mlx-community：Qwen3.5-2B / 4B / 9B（-MLX-4bit）。草稿模型的常规选择是主模型的 1/10 左右 ⇒ **2B 或 4B** 是候选，
  0.8B 太弱猜中率低，9B 太重。Qwen3.6-4B/9B 在 HF 上没有（404）。
- 未做：实测。要么等引擎在批处理路径补上，要么在顺序路径上量一次"没有磁盘 cache 但有投机解码"的 decode 增益作参考。
- **SPD-003 mlx-lm 的投机解码对 Qwen3.5/3.8 全家族不可用**（exp01）：它靠 `trim` 回滚 KV cache，Qwen3.8-27B 48/64 层是 linear
  attention，mlx-lm 用不可裁剪的 `ArraysCache` ⇒ 第一条请求就抛 "requires a trimmable prompt cache"。SPD-002 的候选（Qwen3.5-2B/4B）
  在 mlx-lm 路径上等于没有。顺序路径无草稿 decode 39.2–39.5 tok/s，与批处理路径相同。
- **SPD-004 Qwen3.8 自带草稿头（MTP）**：owner 提醒。原版 `Qwen/Qwen3.8-27B` config `mtp_num_hidden_layers: 1`，第 18 个分片里 15 个
  `mtp.*` 张量（一层 full attention + fc + 三个 norm，共享主模型的 embedding/lm_head），bf16 810 MB。**lmstudio-community 和
  mlx-community 的 MLX 4bit 转换都把它剥掉了**（2180 个张量里 0 个 mtp）。只下那一个分片 + 拆分即可得到草稿器。
- **SPD-005 mlx-vlm（引擎钉住的 321514d = 0.6.12）已经有投机解码和 KV 量化，只是 mlx-engine 没接**：`mlx_vlm/speculative/`
  三种草稿家族（dflash / eagle3 / mtp）、`drafters/qwen3_5_mtp/`（含 `split.py` 拆分工具）、批处理版轮回 `_mtp_rounds_batch`、
  混合注意力回滚 `rollback_speculative_cache`；`kv_quant.py` + `BatchQuantizedKVCache` / turboquant。设计文档 §2 对 B 方案的
  工作量估计（"搬实现"）作废，改为"接线"。
- **SPD-006 实测（exp02，B=1，贪心，mlx-vlm 原样）**：b3 时 story 1.22× / code 1.59× / copyedit 1.70×；接受率高（每轮 2.1–3.0 个）
  但每个草稿 token 折合 0.4 个主模型步（预期 1/11），开销是实现层面的；b2 是散文最优，b≥4 散文倒退。
- **SPD-007 贪心输出不逐字节相同**：code/copyedit 相同，story 第 189 字符起分叉（近平手词被 block 验证的数值差翻转）。进产品要定义
  可接受差异 + 测试（设计文档 §4 验收项要改：不是"逐 token 一致"，而是"代码类逐 token 一致、散文允许分叉且质量不降"）。
- **SPD-008 其他草稿器**：`z-lab/Qwen3.8-27B-DFlash2`（3.6 GB，DFlash v2）存在，这版 mlx-vlm 只写了 v1，未试。z-lab 也有
  `Qwen3.5-122B-A10B-DFlash`（对应 lmk 的 122B 两个版本），未试。
- **SPD-009 投机解码 + 量化 KV 一起开会崩（2026-09-24，exp05）**：mlx-vlm 的校验注意力（`target_verify and L > 1`）对量化 cache 主动返回 None，
  然后按稠密 keys 逐位置切片——上游根本不支持这个组合；两功能各自验收过、合起来第一个请求 500。修在 fork 7a1e17f：量化 cache 时整块
  一次带因果掩码的量化注意力（`kv_quant.verify_block_attention`，经 `patches/qwen3_5.py` 挂钩），稠密路径不动。修后 code/copyedit 与 kv8 普通解码
  逐字节一致、1.49×、采样接受率 86%、itest 5/5。教训：推荐配置按组合验收。
- **SPD-010 投机轮的 cache 快照早了一个 token（2026-09-24，owner 让查的那条警告）**：`[coordinator][WARNING]: Skipping prompt cache save for chunk [0, 256)
  at snapshot 256: quantized kv cache snapshot covers [0, 255), not [0, 256)`，owner 机器上 261 个请求里 4 条，全在 bench 的代码题上（贪心 → 每次落点相同）。
  和 kv8 无关：kv16 + 草稿的临时实例（scratch，port 1236）同一代码题出稠密版 "kv cache snapshot covers [0, 255)"。根因：基类是 decode-ahead，
  `row.tokens` 只记**已喂进 cache** 的 token，快照时 cache 长度 == `len(row.tokens)`；投机轮把还没喂的 bonus（和 `_emit_pending` 里刚采样的 token）
  也 append 了，正好落在 256 的答案就少一个 token，块不存。后果只是命中率（那一块没存），答案不受影响。
  修在 fork e1239e1：先快照、再 append 那个没喂的 token；回归测试让一轮正好落在 256（不修时失败）。修后同题三次 0 条警告。
  顺带：只有 4 条而不是每个 256 边界都有——散文题每次跨过 256（一轮前进多个 token），只有落点恰好等于块边界才触发。
- **SPD-011 DFlash 2 在 M3 Ultra 上不如 MTP 头（2026-09-24，exp07，mlx-vlm 0.6.16 自己的循环）**：27B-4bit 贪心 code 最好 1.26×（4 位草稿器 + 固定块 8），
  story 1.03×，copyedit 1.66×；MTP 头 1.49× / 1.14× / 1.48×。每轮接受确实多（4.3–7.4 对 2.7），但 2B 草稿器一步的**计算**（不是读权重：量化到 4 位不变快）
  把收益吃掉；目标步越快（带宽越高）越吃亏——mlx-dspark 在 M4 Pro 报 2.30× 与此不矛盾（那里普通解码 14.7 tok/s）。三题逐字节一致，采样 code 接受 80%。
  裁：不升级引擎、不接第二种草稿器；这条线留给没有 MTP 头的 Gemma 4。
- **SPD-012 "1.18× 正常吗"（2026-09-24，owner 问；round_bench + exp10）**：不正常，账在校验器。M3 Ultra 上一轮的账：普通单步 29 ms、DFlash2 草稿一步 17.5 ms、
  mlx-vlm 0.6.16 exact verifier 校验 T=4 64.5 ms / T=8 **109 ms**（几乎按 token 线性：它为了位级等价逐 token 干活）。同一台机器 mlx-dspark 一轮 65 ms
  （普通批量前向校验约 47 ms），它每轮接受 3.1（思考文本）所以也只有 1.26×，但用我们的接受率（代码 4.4、块 8 时 4.9–7.4）就是 1.75–2.9×。
  推文的 97 tok/s 在这台机器上没复现（条件不明）。散文每轮只接受 2.0，在 M3 Ultra 上 DFlash2 会比普通解码慢，需要按接受率退回。
  同一个 exact verifier 也是 MTP 头升级后从 1.49× 掉到 1.26× 的原因。出路：校验走普通批量前向 + 记下每层 GDN 的输入（q,k,v,a,b,state,mask,卷积输入）
  喂给 mlx-vlm 自己的 `rollback_speculative_cache`（它有"无中间态就重跑前缀"的分支）——mlx-dspark 就是这么做的（钩 `gated_delta_update` 和 conv 调用）。
  代价：位级等价退回"贪心正确到浮点平手"（mlx-dspark 的定义；0.6.12 的经验是代码一致、散文偶尔分叉）。要 owner 重裁 09-24 的 ①。
- **SPD-013 外部调研：别人在 Apple/MLX 上做到多少（2026-09-24，网上的数，都没在本机验证；M = 来源实测，C = 仅主张）**
  - MTP 头我们不慢：mlx-lm PR #990（M，M4 Pro，Qwen3.5-27B 4bit，temp 0）1.57×；MTPLX（M，M5 Max，Qwen3.8-27B 混合量化）code 58.7 tok/s、
    每深度接受 0.95/0.88/0.80、校验 50–53 ms/轮。我们 M3 Ultra 60.6 tok/s（1.53×）同一水平。
  - DFlash2：oMLX（M，Mac mini M4 24GB，4bit）块 3 1.79× / 块 5 1.66× / **块 8 1.11×**，量化目标固定块 5；mlx-dspark（M，M4 Pro）2.30×，
    但 M3 Max 上与 M4 Pro 同为 33.7 tok/s（作者："wide verify is compute-bound"）。ivanfioravanti 的 97 tok/s（C）只见二手转述，条件不明，且是 **80 核** M3 Ultra（我们 60 核）。
  - llama.cpp Metal 上 MTP 全档变慢（M，M1 Max，Qwen3.5-9B，issue #23752）。
  - 出处：mlx#4265、mlx-lm#990、github.com/youssofal/MTPLX、github.com/ARahim3/mlx-dspark/releases、llama.cpp#23752。
- **SPD-014 Apple GPU 上 4 位校验前向按宽度近似线性变贵——H200 上的倍数不能搬过来**（mlx#4265，M，M3 Ultra 80 核，Qwen3.8-27B 4bit，mlx 0.32.0 + qmv_wide）：
  整模型前向 S=1 29.7 ms、S=4 45.4、S=8 77.1；单个 4 位 matmul M=8 是 M=1 的 6.1×（bf16 平在 2×）。qmv/qmm 切换阈值不是解（已在最优点）。
  已知解是自写小 M 内核（avlp12 `fast_qmm.py`，split-K simdgroup MMA，M 6–8、4 位 gs64，MIT；mlx-dspark 带一份 `small_m_qmm.py`）：S=8 77.1 → **43.3 ms**，
  外挂草稿器 0.92× → 1.65×，逐 token 一致。M<6 不帮忙（MTP 块 3 = 宽 4 用不上）。DFlash 论文的 2.8–6×都假设"校验 8–16 个 ≈ 校验 1 个"（H200）。
  **待核**：我们 exp10 记的"普通批量前向校验约 47 ms"（T 未严格控制）远低于他的 stock S=8 77 ms、接近他带内核的 43 ms——要按依赖链重量一次 T=1..16 的曲线。
- **SPD-015 长上下文下校验的注意力有第二个悬崖**（mlx-dspark，M，M4 Pro）：mlx 融合向量 SDPA 只接 `q_len × GQA ≤ 32`，Qwen3.8-27B GQA 6 ⇒ q ≥ 6 走慢路径，
  32k 时 2.48 → 7.09 ms/次；`sdpa_split.py` 把校验查询切成 ≤5 行一块，8k–32k 快 1.5–2.2×。我们的投机实验 prompt 都短（exp09 最长 4k），agent 的 30k+ 上下文没量过。
- **SPD-016 不靠模型的草稿（后缀/prompt lookup）对 agent 负载是我们没碰过的一块**（C，都不是本机条件）：SuffixDecoding（arXiv 2411.04975，Llama-3.1-8B H100）
  SWE-Bench 2.5×、每步接受 7.8（prompt lookup 3.2），匹配长才用、否则退回模型草稿，草稿长度随匹配长度；Snowflake 代码编辑 1.96–3.12×；mlx-serve（C，M4 Max）
  prompt lookup 复述 2.1×、code 1.5×。草稿几乎零成本（CPU 上查），但长草稿 = 宽校验，撞 SPD-014 的悬崖——和小 M 内核是一对。
- **SPD-017 其余候选与估值（推算，未量）**：按置信度停链 + 按实测校验代价表选 T（SpecDec++ 阈值规则，+7–11% 于 GPU；我们主要赚在散文退回）；
  根部多一个兄弟节点的小树（GDN Tree-Scan，arXiv 2609.23900，Qwen3.6-27B FP8 vLLM B=1，+27% 于 5 步 MTP，收益几乎全来自"第一个草稿被拒"的轮）；
  草稿 lm_head 裁词表（FR-Spec）：248k 词表的 lm_head 约占 MTP 一步读量的 3/4，但一轮里只占约 7%，上限约 5%。不适合本机：lookahead decoding、Saguaro（要富余算力或第二块硬件）。
- **SPD-018 DFlash2 接进引擎（2026-09-24，exp12，fork 3b493b5）**：第二种草稿器 `dflash`（`Drafter.kind`），prompt 分块抓 5 层隐状态只留窗口尾巴（exp09），
  `dflash_round` 用 mlx-vlm 的 `draft_block` / 贪心 walk / 采样 walk，校验走普通前向 + 记录回滚（exp11），单行投机。M3 Ultra 上：code 1.51×（一致）、
  copyedit 1.83×（一致）、story 1.01×（分叉）——与 MTP 头打平（1.53 / 1.51 / 1.20）。kv8 下 code 分叉（量化注意力 L>1 与 L=1 不位级一致）。
  用户面 `model.draft: mtp | dflash2`，缺省按模型页（27B 先 mtp）。DFlash2 的 config `model_type` 写的是目标家族名（"qwen3"），种类看 `architectures` / `dflash_config`。
- **SPD-019 本机校验价目表与一轮的账（2026-09-24，exp13，m3u 60 核，27B-4bit）**：一次前向校验 T 个 token，4k 上下文 T=1/3/5/8/16 = 29 / 40 / 57 / 87 / 149 ms（台阶状）；
  32k 上 T=5 70 ms、**T=6 143 ms**（注意力掉出融合向量 SDPA，比 4k 同宽多 69 ms）。小批量内核（avlp12 fast_qmm）把 4k 的 T=6–8 压到 63 ms、T=9–16 压到 90 ms，
  argmax 全同；但引擎现在的校验宽度是 MTP 3、DFlash2 5（mlx-vlm 的块大小自适应只看接受率，code 约 80% 时停在 5），都不进内核窗口 ⇒ 端到端 +0%，暂不进 fork。
  引擎一轮：MTP 45.7 ms（校验 38.9）、DFlash2 70.9 ms（草稿 15.4 + 校验 54.5 + 回滚 0.7）。加宽后单 token 校验价更低（宽 3 约 13 ms/token → 宽 16 加内核约 5.6），
  所以加宽（后缀草稿的长匹配、DFlash2 块 8）+ 内核 + SDPA 切块是一套，各自单上都不赚。
- **SPD-019 DFlash2 的 4 位版在引擎路径上比 bf16 快（2026-09-24，exp13）**：code 1.71×（67.6 tok/s）、copyedit 2.05×、story 1.18×，接受与一致性同 bf16；
  bf16 是 1.51 / 1.83 / 1.01。exp07 的"4 位不改速度"只在 exact verifier 占大头的循环里成立。M3 Ultra 上 4 位 DFlash2 赢 MTP 头（1.53 / 1.51 / 1.20）于 code 与复述。
  发布为 `seabit-ai/Qwen3.8-27B-DFlash2-4bit`（本机 `~/.cache/lmk-research/qwen3.8-27b-dflash2-4bit`，模型卡与 SHA256SUMS 已备）。
