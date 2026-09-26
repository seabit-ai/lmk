# 带工具的请求为什么不起草（投机解码 × logits processor）

日期 2026-09-25。起因：owner 看 `~/.lmk/logs/lmk.jsonl` 发现 kitten 的 agent 轮（purpose `turn`，每轮都带 `tools`）开投机解码后几乎没有草稿，
bench/smoke（不带工具）照常起草。编号接 `research/2026-09-23-speculative-decoding` 的 SPD-021。

## 目的
1. 证实或证伪："带工具 ⇒ 引擎挂上工具/思考守卫 logits processor ⇒ `_can_round()` 拒绝投机轮"。
2. 摸清守卫每个 token 做什么、存什么状态，哪些是无状态掩码、哪些是有状态机。
3. 给出让带工具请求重新起草的 2–3 个具体做法（fork 里改什么、风险、怎么验）。只调研，不改代码。

引擎 = fork `seabit-ai/mlx-engine` 分支 `lmk`，`ENGINE_COMMIT` 3b493b5（下文 fork 路径都相对 `.engine/mlx-engine/`）。

## 发现

- **SPD-022 假设成立：带工具的 Qwen3.8 请求一定挂上守卫，挂上守卫的行一定走普通解码。** 路径：`generate.py:437-438` → `_batched_generation`
  → `generate.py:734-781`（批处理视觉路径、没有 `json_schema` 时依次试 Muse Glimmer / Gemma4 / Qwen3.5 三种工具上下文，命中就 append 对应守卫）
  → `model_kit.generate(logits_processors=...)`（`generate.py:783-793`）→ 行的 `logits_processors` 非空 → `speculative.py:460`
  `if any(... or row.logits_processors for row in self._rows): return False`。Qwen 这支的触发条件（`tool_runtime.py:73-106`）：`model_type` 以
  `qwen3_5` 开头、tokenizer 的工具起止标记是单 token `<tool_call>`/`</tool_call>`、渲染后的 prompt 含 `<tool_call>` 与 `<function=`，
  且 `<tools>…</tools>` 里能解析出至少一个工具名——只要请求带 `tools`，Qwen3.8 模板渲染出来就全满足。
  实测（scratch 脚本，只加载 tokenizer，`lmstudio-community/Qwen3.8-27B-MLX-4bit`，一个 `read_file` 工具）：带工具 → `Qwen35ToolContext`，
  不带工具 → `None`。没有其它处理器会挂：`repetition_penalty` 只在 ≠1.0 时挂（`utils/generation_helpers.py:23-48`），turn 的 `sampling` 字段只有
  temp/top_p/top_k。把握：高（读码 + tokenizer 实测）。

- **SPD-023 日志统计复算：113 个带草稿字段的 turn，1 个 `draftDrafted>0`；那一个是别的请求漏进来的计数。** 数据：`~/.lmk/logs/lmk.jsonl`
  截至 2026-09-25 19:03，`LmkChatDone` 464 条，带 `draftDrafted` 字段且 purpose=turn 的 113 条（owner 数的 114 差一条，可能是统计时点不同），
  全部在同一配置下（`qwen3.8-27b-4bit`、`draft: dflash2`、`thinking: false`）；112 条 drafted=0。按时间窗看，112 条与任何别的请求都不重叠，
  照样 0——排除"两行并发所以不投机"（`MAX_ROUND_ROWS = 1`）。唯一 drafted=1/accepted=0 的那条（refId `s-1790321690769/t-mugphl58.1`）
  开始于 1790325363574，第二个 `research` 请求结束于 …363646；两个 research 分别记 drafted 2/accepted 2 与 3/2，差正好是 1/0 = 这条 turn 记的数。
  原因：草稿计数是草稿器模型上的全局累计，lmk 按"生成开始前后相减"归给请求（`lmk/engine.py:243-252`），并发期间会串账。
  附带证据：`compaction`（kitten 也带工具，`kawa/compaction.go:353` 传 `face`）独占运行 700 token，drafted=0。
  **日志里没有"这次请求带没带工具"的字段**，"turn 都带工具"来自 kitten 源码（`kawa/event.go:821` `Tools: toolDefs(face)`，registry 总非空），
  不是日志。不带工具的 turn：日志无法区分，源码上不存在。把握：高（统计）/中（串账解释，时间与差值吻合，未复现）。

- **SPD-024 思考开/关都挂守卫；Gemma 4 同样挂，但 lmk 的 Gemma 没有草稿器，不受影响。** 守卫是否挂上只看工具，不看思考；思考只决定初始
  `reasoning_open`：Qwen 关思考时 prompt 以 `<think>\n\n</think>\n\n` 结尾 → `False`，开思考以 `<think>\n` 结尾 → `True`
  （`tool_protocols.py:45-46`，tokenizer 实测）。Gemma 4（26B-A4B、31B，`model_type gemma4`）带工具 → `Gemma4ToolContext`（`tool_runtime.py:52-70`），
  两种思考设置下 `reasoning_open` 都是 False（开思考时 prompt 不预置 `<|channel>thought`，由模型自己决定）。`lmk/models.py` 里只有 Qwen3.8-27B
  的几档有 `drafts`，所以投机×工具的问题当前只落在 27B。把握：高。

- **SPD-025 守卫是一个对象三层职责：思考段屏蔽工具起始（无状态掩码 + 一个布尔）、工具调用体走 llguidance 语法（有状态）、调用结束后的出口车道（小状态机）。**
  `NativeToolReasoningGuardLogitsProcessor`（`tool_runtime.py:229-461`），Qwen 与 Gemma 子类只是改名（:464-469）。状态：
  `_reasoning_open_mx`（mx 布尔，看到 `<think>` 置真、`</think>` 置假；Gemma 的开标记是两 token，所以还存 `_previous_token_mx`）、
  `_tool_state` ∈ NORMAL/TOOL/POST_TOOL、`_tool_matcher`（llguidance `LLMatcher`，进 TOOL 时新建）、`_context_token_count`（追赶游标）。
  每 token（`_process_last_token_mx`，:363-461）：
  ① 用刚喂进的 token 更新思考布尔；② NORMAL：若刚喂的是 `<tool_call>`，只放行 `"\n"`（语法的首 token，:406-410），否则不动；
  ③ TOOL：把 token 喂给 matcher（这里 `.item()` 同步一次），语法完成 → POST_TOOL 并只放行 EOS/空白/`<tool_call>`（:422-426），否则用 matcher 的 bitmask 掩码（:430）；
  ④ POST_TOOL：只放行 EOS/空白/下一个 `<tool_call>`（:432-451）；⑤ 思考开着时把 `<tool_call>` 置 −inf（:453-460）。
  进 TOOL 不是在 ② 里，而是**下一次调用的追赶循环**看到 context 末尾是 `<tool_call>` 时（:301-319）。Prefill 时 `__call__`（:277-292）按 prompt 重置。
  所以：**Qwen 守卫确实用语法约束调用体**（不只是思考期间挡一个 token）：`_qwen35_llguidance_grammar`（:593-608）锁住工具名只能是声明过的、
  `<function=…>`/`<parameter=…>`/`</parameter>`/`</function>`/`</tool_call>` 的结构，参数值是任意文本但不得含闭合标签。
  逐 token 模拟（关思考，`read_file` 一次调用）：普通正文每步放行全部 248,320 个 token（等于恒等）；`<tool_call>` 之后放行 1 个；
  体内结构位置放行 1–427 个，参数值位置 247,868 个；`</tool_call>` 后 6 个。开思考时思考段放行 248,319（只挡 `<tool_call>`）。
  每步开销 0.4–3.7 ms；每次进入工具调用新建 matcher 约 52 ms（进程里第一次还要建 llguidance tokenizer，约 1 s，之后缓存）。把握：高（读码 + 模拟）。

- **SPD-026 今天的投机轮：草稿 → 一次批量前向校验 → 按采样器走接受前缀 → 按接受数回滚。** `SpeculativeGenerationBatch.next`（`speculative.py:464-474`）：
  能投机就先 `_emit_pending`（:542-569，把上一步采样出、还没喂的 token 发出去并 append 进 `row.tokens`，它成为本轮的 bonus），再 `_round`（:571-640）。
  DFlash（当前配置）：`dflash_round`（:295-331）草稿一块 → `_verify_block`（:164-191）对 `[bonus, drafts…]` 一次普通前向，`GdnVerifyRecorder`
  记下线性注意力的输入 → 贪心走 `_speculative_walk`（逐位比 argmax），采样（turn 是 temp 1.0/top_p 0.95/top_k 20）走 mlx-vlm 的
  `_sample_dflash_target_walk`：每个位置用行的采样器从目标 logits 抽一个，与草稿相等就继续，不等就收下目标的那个并停（这就是"抽样比对"，分布与逐 token
  解码一致）→ `accepted < block-1` 时 `rollback_speculative_cache`。MTP 行：`speculative_round`（:202-292）贪心用隐状态上的融合 argmax（`skip_logits=True`，
  **不出 logits**），采样行用 `_speculative_walk_batch_deferred_greedy`。**整轮没有任何地方调用 logits processor**；轮末最后一个 token 是下一轮的 bonus，
  不进 cache。把握：高（读码）。

- **SPD-027 让一轮尊重处理器，不需要回滚处理器：按位置顺序"先处理、再抽样、比对、不等就停"，处理器只吃被接受的 token。**
  校验前向的第 j 个位置正好对应普通解码的一步："喂进 `verify_input[j]`、得到下一个 token 的 logits"。所以对 j = 0,1,…：
  `logits_j' = proc(context = 本轮前的历史 + verify_input[:j], last = verify_input[j], logits_j)` → 用行的采样器从 `logits_j'` 抽 `t_j` →
  若 j 还在草稿内且 `t_j == draft_j` 继续，否则收下 `t_j` 并停。处理器被喂的只有 bonus 与被接受的草稿，全都真的发出去了；被拒草稿从没进过处理器，
  也就没有要回滚的处理器状态；`t_k`（纠正 token）成为下一轮的 bonus，下一轮第 0 位才喂给处理器——与普通解码的"喂进才更新"完全同构。
  目标 cache 的回滚照旧按接受数。贪心时输出与"处理器 + 逐 token argmax"逐位相同（外加 exp11 已接受的浮点平手差异）；采样时分布相同。
  这正是 llama.cpp 的做法：`common_sampler_sample_and_accept_n`（`common/sampling.cpp`，WebFetch 2026-09-25 读到）逐位 `sample`（带语法）→ `accept`
  进语法 → `draft[i] != id` 就 break，**没有语法回滚**。vLLM 是另一路：先 `validate_tokens` 把草稿截到合语法的最长前缀，为每个投机位置预算一行 bitmask
  （推进 matcher 再 rollback），再并行做拒绝采样——为 GPU 上多行并行设计；SGLang 对草稿树 DFS 逐节点 accept/填掩码/回滚（SO-019，
  `structured-output` 分支的 `research/2026-09-25-structured-output/notes.md`）。本引擎投机只跑单行（`MAX_ROUND_ROWS = 1`），DFlash 的采样 walk
  本来就是逐位 `mx.eval` 的 Python 循环，**llama.cpp 式最贴合、改动最小**；vLLM 式的"预算全部掩码 + rollback"在这里只多出一套状态回退。
  需要的改动点：① MTP 行要有每位置 logits（现在 `skip_logits=True`，要对块内隐状态过一次 lm_head；DFlash 已有 `out.logits`）；
  ② 引擎自己的 walk 替掉 mlx-vlm 的 `_speculative_walk*`/`_sample_dflash_target_walk`（只在行带处理器时走）；③ `_can_round` 放开处理器条件
  （top_logprobs 仍挡）；④ 修 SPD-028 的交接问题。每位置多花 0.4–3.7 ms（SPD-025），DFlash 一轮 70.9 ms（SPD-019）、块宽 5 ⇒ 最多约 +2–18 ms/轮。
  把握：高（机理，读码）/中（成本，未在轮里量）。

- **SPD-028 现有"轮 ↔ 普通步"交接对处理器不安全，任何方案都得先修它。** 普通步里 `row.tokens` = 已喂的 token，采样出的那个等下一步喂进时才 append
  （decode-ahead，`batch_generator.py:485-506`，处理器拿 `row.tokens` 当 context、`inputs` 当 last token）。但一轮之后 bonus 已经 append 进 `row.tokens`
  （`speculative.py:626`，`_emit_pending` 也是 :559），接着的 `_plain_tick`（:476-521）又把同一个 bonus 当 `inputs` 喂——守卫的追赶循环会把 bonus 当 context
  吃一次、再当 last token 吃一次：在 TOOL 状态下 llguidance 会收到同一 token 两次（`consume_token` 报错 → `ValueError`，:213-217）；
  通用处理器那支（`_apply_logits_processors` :274-278）还会把 last token 再 append 进 `row.tokens`。今天碰不到，因为带处理器的行从不进轮。
  另外守卫的追赶循环**只追工具状态、不追思考状态**（:301-319 只看 TOOL/`<tool_call>`；思考布尔只在 `_process_last_token_mx` 里按 last token 更新）——
  所以"跳过处理器跑几轮、事后补"是错的，每个发出的 token 都必须按顺序当 last token 喂一次。把握：高（读码，未构造失败用例）。

- **SPD-029 便宜的特例只覆盖一半：思考/正文段守卫是"由已喂前缀决定的无状态掩码"，可以整块向量化；工具调用体是 llguidance，必须逐位。**
  NORMAL 状态下（关思考的 turn 里除了调用体都是它），守卫对位置 j 的效果只取决于 `verify_input[:j+1]`：思考是否开着（扫 `<think>`/`</think>`）、
  `verify_input[j]` 是否 `<tool_call>`。所以"块内扫一遍标记、对思考开着的位置挡 `<tool_call>`、在第一个 `<tool_call>` 处截断本轮"就能让正文与思考段照常起草，
  不碰 llguidance；进了调用体（TOOL/POST_TOOL）就退回普通步。但注意不能"草稿里出现被挡 token 就拒掉"了事：采样行的目标 token 本身可能抽到
  被挡的 `<tool_call>`，屏蔽要施加在抽样之前（改变归一化），所以仍要在 walk 里对 logits 施掩码，只是掩码能一次算完。
  代价：kitten 的 turn 大多以工具调用收尾（113 条里 91 条 `toolCalls>0`），而调用体（写文件内容、命令）恰是草稿接受率最高的复述型文本（exp12 copyedit 1.83×）——
  这个特例把最赚的那段放弃了。另：`<tool_call>` 之后的下一 token 只允许 `"\n"`，如果 `<tool_call>` 作为纠正 token 发出，下一轮必须是普通步。把握：高（机理）/中（"调用体占大头"未量 token 比例）。

- **SPD-030 用户感受得到的量：turn 很短，单轮能省的秒数有限。** 113 条 turn 共 5,937 个 completion token，中位 30、最多 469。按普通解码 39 tok/s 与
  DFlash2 在引擎路径的 1.2–1.8×（SPD-020）推算，平均每轮省约 0.2–0.6 s，长回复（写文件，400+ token）省 3–5 s。推算，未量。把握：中。

- **SPD-031 fork 里没有覆盖"投机 + 处理器"的测试。** `tests/test_batched_vision_speculative.py` 所有批次都用 `logits_processors=[[]]`（:91、:177、:297），
  "让步退回普通"的测试只测 top_logprobs 与 opt-out（:145-152），没有"带处理器 ⇒ 普通步"的断言；`tests/test_tool_runtime_reasoning_guard.py`
  全是单处理器逐 token 的测试，不涉及投机。把握：高。

## 设计选项

**A. llama.cpp 式：walk 里逐位处理、抽样、比对（推荐）。**
- fork 改动：`speculative.py` 加引擎自己的 `_processed_walk(logits[1,T,V], verify_input, history, processors, sampler)`（SPD-027）；
  带处理器的行走它，不带的照旧走 mlx-vlm 的 walk；MTP 行在带处理器时对块内隐状态过 lm_head 拿 logits；`_can_round` 去掉处理器条件；
  修 SPD-028（给处理器的 context 一律是"已喂的 token"，round 与 plain tick 同一口径）。同一套对 `json_schema` 处理器（结构化输出裁决六的 backlog 项）直接适用。
- 风险：交接口径（SPD-028）错一位就是语法双吃 → 500；每位置一次 Python 同步 + 掩码，窄块时可能吃掉部分收益；MTP 的 lm_head 额外开销未量。
- 验证：引擎单测——假模型 + 真守卫（Qwen tokenizer 可离线）构造"正文 → `<tool_call>` → 调用体 → `</tool_call>`"，对比"逐 token + 处理器"与"轮 + 处理器"
  的 token 序列相同；草稿在调用体里给一个被语法屏蔽的 token，断言接受在该位停；round 后接 plain tick 不报错。itest（exp03/exp12 的做法）：同一带工具的
  prompt，贪心 400 token，开/关草稿 sha 逐字节对照（kv16；kv8 已知 code 会因量化注意力分叉，单列），再在 owner 机上 kitten 跑几轮看 `draftDrafted>0`。

**B. 特例：只在 NORMAL 段起草，调用体退回普通步。**
- fork 改动：守卫加一个"块掩码"方法（按 `verify_input` 扫思考/`<tool_call>` 标记，SPD-029），walk 前一次性施加；本轮截在第一个 `<tool_call>`；
  `_can_round` 只在守卫处于 NORMAL 时放行；轮后把发出的 token 按序喂给守卫以同步思考布尔；同样要修 SPD-028。不碰 llguidance。
- 风险：改动面小于 A，但仍要改 walk（掩码要在抽样前）；收益只在正文/思考段，turn 最赚的调用体不起草；两条路径（掩码块 / 逐位）以后接 `json_schema` 还得再做一次。
- 验证：同 A 的单测（去掉调用体内起草的那条），加"调用体期间 `_can_round()` 为 False"；itest 同 A 的 sha 对照，外加统计调用体内 drafted=0、正文段 >0。

**C. vLLM 式：处理器提供"整块掩码 + 回滚"接口，先截草稿再并行校验。**
- fork 改动：守卫加 `block_masks(verify_input) -> [T,V]`（在 matcher 上 `consume` 草稿再 `rollback`，llguidance 1.8 有 `rollback`/`validate_tokens`）与
  `commit(accepted)`；草稿先按 `validate_tokens` 截断；walk 用预算好的掩码。为多行并行准备。
- 风险：处理器多一套"试探再撤回"状态，思考布尔等 mx 状态也要能撤；与 A 等价的正确性却多一处出错面；今天单行投机用不上它的并行好处（SPD-021 多行仍不对）。
- 验证：同 A，再加"每个位置的块掩码 = 逐 token 推进时的掩码"的单测，以及 rollback 后 matcher 状态与未试探时相同。

- **SPD-032 在轮里结束的请求交出的热 cache 比它报的 all_tokens 少一个 token；下一轮带 tools 的对话整段命中它（自 b58a72e 起，e053a72 让 agent 轮都碰上）。**
  轮的最后一个 token（以及 `_emit_pending` 发出的待喂 token）按设计不进 cache，但 `_response` 交出 `all_tokens=list(row.tokens)` 含它；热 entry 以
  `all_tokens` 为"cache 覆盖的 token"，下一请求整段前缀命中（`coordinator._load_hot_restore_plan`，trim 0）时多报一个。实测（exp04，27B-4bit kv8 dflash2，
  思考开 + low，fizzbuzz 工具任务）：热 entry 807 token、KV offset 806，缺的是 `<|im_end|>`；Qwen 模板重渲染第一轮与生成的逐 token 相同，第二轮整段命中，
  引擎报 cached 807。净效应在这一例里小：第二轮首位置 logprob 差 ≤ 0.25，60 token 文本相同——没有观察到"乱码"，但位置错一位、少一个 token 是真的，
  长对话里会一轮轮叠加（每轮少一个结束标记）。修：`all_tokens` 不含未喂的那个 token（`_response`），而不是交出前再喂一次——前者不多一次前向、
  不动草稿器状态，代价是下一轮多 prefill 1 个 token；命中数与 cache 实际覆盖一致。fork 42a248c。把握：高（读码 + 单测红→绿 + 真机 offset 对照）。
- **SPD-033 DFlash 轮按 walk 的接受数回滚、在截断之前：草稿里接受了停止 token 之后的 token 时，结束行的 cache 比 all_tokens 长。** MTP 轮本来就先截断再按
  `len(cut)-1` 回滚；DFlash 改成同一口径（42a248c）。单测：`max_tokens` 落在块中间（1/7/10/13）与停在块中的 EOS，两种草稿器都断言 `all_tokens == 目标实际喂过的 token`。
  另：带 processor 的 walk 遇到停止 token 即停（06f73c1），不再把 EOS 之后的草稿喂给 processor。把握：高（单测）。
- **SPD-034 kv8 上热 cache 续算与"磁盘恢复 + 重新 prefill"给出的 logits 明显不同（首位置 logprob 差到 2.75），修复 SPD-032 之后仍在。** exp04：同一第二轮 prompt，
  A 热 cache（806 命中，prefill 26）对 B 磁盘恢复（512 命中，prefill 320），两次文本相同但 top-5 logprob 差 2.75–3.0。与 exp01–03 "关草稿冷/热跑 sha 不同"
  可能同源（kv8 下解码/校验逐步写进的量化 KV 与整块 prefill 写进的不同）。未查。把握：高（数）/低（原因）。
