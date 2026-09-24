# 引擎路线：fork mlx-engine，先量后建

> 2026-09-23 SAD 开题。owner 提出"我们自己写一个 mlx-engine"；我的判断是"目标同意，路径是 fork + 补丁，不是从零写"。
> 本文记已摆出的事实、双方立场、裁定。09-23 SAD 三点已裁（fork 放哪、验收、用户面）；**开工令另等 owner 显式给**。
> 取证：`research/2026-09-23-speculative-decoding/`（SPD-001/002）。

## 0. owner 要的结果
lmk 的引擎路线图归自己：**投机解码与磁盘 cache 同时有，KV 量化**——owner 的原话："those are already critical enough"。

## 1. 已摆出的事实
- 栈：lmk → mlx-engine（LM Studio，1.3 万行 Python，调度层，不含模型定义）→ mlx-vlm（模型定义，个人项目 Blaizzy/Prince Canuma）
  与 mlx-lm（苹果 ml-explore；lmk 只用它的分词器加载与工具调用解析器）→ MLX（苹果，C++ + Metal 内核，Python 只是绑定）。
- mlx-engine 两条路径：老 `ModelKit`（顺序、内存 cache、**有投机解码**）与新 `BatchedVisionModelKit`（连续批处理、**磁盘 cache**，
  2026-07 加入，5,500 行）。新路径对投机解码与 KV 量化都直接 raise "not supported"。**二选一是同一引擎两代代码各缺一块，不是原理互斥。**
- mlx-engine 单一主要作者（111/195 提交），最近提交 2026-08-21——上游节奏是真实风险。
- lmk 已经在做的事等于"轻 fork"：锁提交号 + 换掉 cache 存储类。
- ~~投机解码兼容判据 = 词表相同；Qwen3.8-27B 的草稿候选：Qwen3.5-2B / 4B~~ **作废**（09-23 exp01，SPD-003）：mlx-lm 的投机解码
  对 Qwen3.5/3.8 的混合注意力不可用（cache 不可裁剪）。
- **mlx-vlm 在引擎钉住的版本里已经实现了两件事**（09-23，SPD-005）：投机解码（dflash / eagle3 / mtp 三种草稿器，含批处理轮回和
  混合注意力回滚）和 KV 量化（批处理 cache，uniform / turboquant）。mlx-engine 的批处理路径只是没接。
- **Qwen3.8 自带一层 MTP 草稿头**（SPD-004），MLX 转换都剥掉了；下原版一个分片、用 mlx-vlm 的拆分工具即得 810 MB 草稿器。
  实测（SPD-006）B=1 贪心：散文 1.22×、代码 1.59×、复述 1.70×；贪心输出散文不逐字节相同（SPD-007）。

## 2. 两个方案
| | A. 从零写引擎 | B. fork mlx-engine，在批处理路径上补两项 |
|---|---|---|
| 投机解码进批处理路径 | 全部重写 | ~~搬老路径的实现，估一两千行~~ **接线**：mlx-vlm 已有批处理版 MTP 轮回和混合注意力回滚；要做的是 BatchedVisionModelKit 加载草稿器、批生成循环改走 `run_speculative_rounds`、磁盘 cache 块与回滚对齐（09-23 修正） |
| KV 量化 | 全部重写 | ~~接 mlx-lm 的量化 KV cache~~ **接线**：mlx-vlm 已有 `BatchQuantizedKVCache`/turboquant；要做的是引擎批处理路径传 `kv_bits`、拟合公式认量化后的每 token 字节、磁盘存取认量化数组 |
| 拟合公式、调度、图片跨块 | 重来，且 3 GiB 预留那类校准值要重测 | 沿用 |
| 底层 | 仍是 mlx-vlm | 仍是 mlx-vlm |
| 上游死了 | 无影响 | fork 即我们的引擎 |
| 上游活着 | 白做 | 补丁可回合 / 上游追平就删补丁 |
我的立场：**B**。A 只在"架构做不到我们要的事"（按请求思考开关、多模型常驻）或"上游停更半年以上"时才值得；两者都没到。

## 3. 先量后建（已量，2026-09-23）
原计划：老路径上 27B-4bit + Qwen3.5-4B 草稿。**走不通**（exp01：mlx-lm 不支持混合注意力回滚）。改量 mlx-vlm 的原生 MTP 草稿（exp02）：
- 判据：≥ 1.8× 投机解码先做；≤ 1.3× KV 量化先做；中间开会。
- 实得（B=1、贪心、mlx-vlm 原样、block 3）：**散文 1.22×，代码 1.59×，复述 1.70×**——灰区。接受率高（每轮 2.1–3.0 个），
  拖后腿的是每个草稿 token 折合 0.4 个主模型步的实现开销（预期 1/11）。压掉这块开销是 fork 之后的优化项。
- 判据设立时的前提变了：两件事都不用"实现"，只用"接线"（§2 修正）。先做哪个不再由工作量定，由用户价值定——见 §4。
详见 `research/2026-09-23-speculative-decoding/exp02-27b-native-mtp/results.md`。

## 4. 待裁的点（SAD 续场）
- **fork 放哪——已裁（09-23，owner：A）**：GitHub fork `seabit-ai/mlx-engine`，在 08f0c07 上开 `lmk` 分支。lmk 只换 install.sh 的 tarball URL 和
  Makefile 的 clone URL，ENGINE_COMMIT 仍是 commit 号，cache-fixture/cache-compat 升级流程不变。补丁按功能一个 commit（KV 量化一个、投机解码一个），
  上游一动就 rebase 到新的上游 commit，冲突在 rebase 时解；上游已一个月零提交，此成本目前接近零，上游大改再议长期分支。
  否决 B（vendor 进 lmk 仓库）：它把"能否跟上游"从 git 操作变成人工 diff，而跟上游正是选 fork 不选重写的全部价值；且 13k 行引擎会盖过几千行的产品代码。
- 先做哪个：§3 的数落在灰区，且两件都是接线。我的倾向：**KV 量化先**——收益确定（32 GB 档同内存两到四倍上下文，直接改
  README 那张表的 ctx 列），无输出差异问题；投机解码 1.2–1.7× 且散文输出会变，还要先解决"lmk pull 怎么带草稿器"（MLX 转换里
  没有，要自己拆或自己发布一份）。待 owner 裁。
- **KV 量化的验收——已裁（09-23，owner 认）**，对着客户会撞上的四种坏事：
  1. 答案变笨（长上下文才显）：现有 15 道代码/工具题加一臂，题前垫 6 万 token 真实内容，bf16 对 8 位机器判分，不降出噪声范围。唯一新写的测试。
  2. 表里 ctx 数字假：拟合公式的每 token 字节由探针用量化 cache 实测（不手算 53%），96 GB 上实测真实上限进 `measured_context_on_96gb`，README 表单测照旧。
  3. 磁盘 cache 出事：cache 身份键带位宽，bf16 写的块在量化会话下干净跳过（单测）；8 位写的块重启后还原（cache-fixture / cache-compat）。
  4. 变慢：bench 三个数按条件并列记进 benchmarks 表，只记录不设门槛。
  不写"输出逐字节一致"——量化 KV 本来就改数值。门槛：模型页写上 8 位推荐之前，1、3 必过，2 必实测；与模型进表是同一根尺子（itest 开关各 5/5 且 agent 任务全对）加长上下文一臂。
- 投机解码的验收（未裁，等做到它时再细）：一致性不能写成"逐 token 一致"（SPD-007：散文会分叉）；改为"代码/复述类逐 token 一致，散文类允许分叉但评测判分不降"。
- **KV 量化的用户面——已裁（09-23，owner 认；命名依据 `research/2026-09-23-kv-cache-naming/`）**：
  - 配置项 `model.kv_cache_bits`，整数，只认 16 / 8 / 4，别的值报错说清楚；K/V 分开时另加一行 `kv_cache_value_bits`，不做 `k8v4` 字符串。
    注释按 Ollama 的三个量写：省多少、代价多大、对照物是什么。否决新造词：十家框架无一家发明用户友好词，客户搜的就是 "KV cache"。
  - 默认 16（不开），与全行业一致；过了验收门槛的模型在模型页"推荐配置"里写 `kv_cache_bits: 8`——推荐按模型，开关归客户，
    与 27B 页的 `reasoning_effort: low` 同一机制。否决默认全局开（替没测过的模型下结论，且让老用户的磁盘 cache 一夜失效）；
    否决按 Mac 内存档自动选位宽（客户看不见的魔法）。
  - README 表的 ctx 列显示**推荐配置下**的数，格子加脚注指向模型页；未推荐的模型仍显示 16 位的数。`TestedModel` 加 `kv_cache_bits` 字段，
    单测锁表照旧。否决一格两数（给工程师看的）。
  - k8v4：mlx-vlm 批处理路径的 uniform cache 只有一个 `bits`（K、V 同宽），分开要改 mlx-vlm 约十行——第二个补丁对象；
    turboquant 路径已支持分开但质量无数据。做法：长上下文评测臂多跑一个 k8v4 变体，有数再决定值不值得多背一个补丁。
- **落地状态（09-23）**：fork 建好（owner），分支 `lmk` 上 KV 量化接线 d3650db；lmk 分支 `kv-cache-bits`。验收 2/3/4 过，
  itest 8 位开关各 4/4 + 重启还原过；**第 1 条（长上下文评测臂）未做**，做完才写模型页推荐。数据 `research/2026-09-23-kv-cache-quant/`。
- **投机解码 SAD（09-23，四点全部已裁）**：
  1. 草稿器从哪来——**已裁 B**：Seabit 在 HF 发拆好的草稿器（`seabit-ai/Qwen3.8-27B-MTP-draft`，Apache 2.0 允许再分发），`lmk pull` 直接下。
     A（lmk pull 现场下原版分片拆）留作兜底；否决 C（让用户自己拆：把最难的一步推给用户）。**卡点：HF 上还没有 `seabit-ai` 组织**，
     owner 建组织并把 Xinkai569 加进去后，上传/模型卡/校验和归我（我这边有该账号的 write token）。
  2. 用户面——**已裁**（命名依据 `research/2026-09-23-spec-decode-naming/`）：布尔开关 `model.speculative_decoding: true|false`（默认 false，
     与 `thinking` 同样式；模型页推荐时写 true），高级项 `model.draft_tokens`（每轮猜几个，默认模型页的数，Qwen3.8-27B 为 3）。
     否决 `draft_model: auto|off`（我把开关和将来的仓名塞进一个键；且 YAML 把 `on`/`off` 解析成布尔，一个键混两种类型是 `id`/`name` 那种乱）。
     将来外挂草稿器另加 `draft_model: <HF 仓或路径>`，MVP 不要。**开在没有草稿器的模型上 = 启动时干净报错退出**（fail fast，
     走 `LmkConfigInvalid` 那条路，不是静默退回普通 decode）。状态行 `· draft on` + 自启动以来接受率；usage chunk 加接受的草稿 token 数；bench 加一行接受率。
  3. 验收——**已裁**（"sounds about right"）：代码/复述类贪心逐 token 一致；散文允许分叉但评测判分不降；另量两条进记录不进门槛：
     采样开着（Qwen 默认 temp 1.0）的接受率、并发 2 时的收益。
  4. 范围——**已裁**：MVP 先做 Qwen3.8-27B 的自带 MTP 头，其他草稿家族（z-lab DFlash 给 122B、Gemma assistant/DFlash）之后加；
     **只要可能就支持批量**（mlx-vlm 有 `_mtp_rounds_batch`），不做"并发就退回普通 decode"的第一版。
- 草稿器的发行：lmk 的模型是 `lmk pull` 的那份下载；草稿器是我们从原版权重拆出来的，要么 lmk pull 顺手多下一个分片现场拆
  （3 GB 下载换 810 MB），要么 Seabit 在 HF 发一份拆好的。

## 5. 投机解码接线方案（09-23，读完 mlx-vlm 与引擎解码循环后定；实现细节，不是 SAD 点）
**事实**
- mlx-vlm 的 `_mtp_rounds_batch` 是一个封闭的生成器：自己管一批行的整个生命周期（行结束就 filter），**不接受中途加行**。引擎的
  `GenerationBatch` 是"一步一 token"的循环，靠 `append_prefilled_sequence` 在两步之间把新 prefill 完的行并进来（连续批处理）。
  两者不能直接套；要把 mlx-vlm 的"一轮"拆出来当引擎的"一步"。
- Qwen 的 MTP 草稿器（`Qwen3_5MTPDraftModel`）**不用** `shared_kv_states`（`set_shared_kv` 只记位置），自己有一层 KV cache，
  靠 `accept_verified_tokens_batch` 用验证时的目标 hidden 逐 token 增长。B=1 路径还会用 `prefill_from_target_hidden` 把整个 prompt
  过一遍草稿层（exp02 量的是这条）；批量路径不做这步，草稿器从空 cache 起步。引擎接批量路径的做法（prompt 分块 prefill、磁盘还原都
  没有整段 prompt 的 hidden），接受率可能比 exp02 低——要量。
- 目标模型（qwen3_5 language.py）有 `speculative_argmax_from_hidden`（贪心验证不算整张 logits）和 `rollback_speculative_cache`
  （48 层 linear attention 的状态回滚靠验证时返回的 `gdn_states`）。
- 引擎的 `_step` 是"先出上一步采的 token，再算下一步"（decode-ahead）；每行有自己的 sampler、logits processors、top_logprobs；
  `next()` 每行每步恰好一个 `Response`。

**做法：`SpeculativeGenerationBatch`（引擎新文件 `batched_vision/speculative.py`），一步 = 一轮**
1. 起步：`_PromptPrefill.generate` 的最后一遍前向加 `return_hidden=True`，把最后一个 prompt token 的 hidden 存到批上（`_next_hidden`），
   连同已采样的首 token（bonus）。
2. 一步：草稿 `_mtp_draft_block_active` → 验证 `_mtp_verify_target`（贪心走 argmax-from-hidden）→ 逐行 walk（贪心 `_speculative_walk_batch`；
   采样时按行各用自己的 sampler 做 deferred walk）→ **先按 stop / max_tokens 截断每行的新 token 列表**（截断后的接受数再喂回滚，cache
   才不会多出没吐出的 token）→ 草稿器 `accept_verified_tokens_batch` → 目标 `rollback_speculative_cache` → 每行取接受位置的 hidden
   作下一轮输入 → 位置 += 接受数 + 1 → `set_shared_kv({}, …)` 重绑位置。
3. 出 token：每行每步 0..bs 个，`next()` 改为每行每步多个 `Response`（顺序、逐个过 stop 判定），首步前面带上 bonus。
   logprob 不可得（验证不算 logits）：写 0.0 并在文档里说明；**要 top_logprobs 或带 logits processors（重复惩罚）的行进批时，整批这一步
   退回普通 `_step`**——按步判定，不是按请求拒绝。带图片（`rope_deltas`）的批同样退回普通步（验证调用不带 mRoPE）。MVP 只做 Qwen3.5/3.8 家族文本。
4. 加行：`append_prefilled_sequence` 时 hidden 与 bonus 拼接，**草稿器整体 `reset`**（它的 cache 只装生成过的 token，重来只损失几轮接受率）；
   行结束用 `filter_batch`。不去拼草稿器的内部数组（那是 mlx-vlm 的私有布局）。
5. 磁盘 cache 快照（`_emit_cache_save_snapshot`）每步每行调一次，按 `row.tokens` 长度判块边界，一步最多 bs 个 token，跨不过一个 256 块。
6. 加载：`BatchedVisionModelKit.load_draft_model(path)` 用 mlx-vlm 的 `load_drafter`，`is_draft_model_compatible` = 草稿器 `model_type`
   是 `qwen3_5_mtp` 且 `text_config.hidden_size` 等于目标；`generate(...)` 的 `speculative_decoding_toggle` / `num_draft_tokens` 照老路径的语义。
7. 验证：引擎侧单测只测账目（多 token 出列、stop 截断、退回普通步的判定），用假的"一轮"函数；真正的验收在 27B 上：贪心代码逐 token 一致、
   bench 的 decode 与接受率、采样默认值下的接受率、并发 2 的收益——都是 research 记录。

**没解的**：每个草稿 token 0.4 步的开销（profile 后再说）；Gemma 家族（草稿器要 shared_kv，且 `_mtp_rounds_batch` 的 shared_kv 切片逻辑
要搬过来）；外挂 DFlash 草稿器（122B）。
