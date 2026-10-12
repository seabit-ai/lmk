# 裁决 A 实现报告（spec-with-tools）

日期 2026-09-25。状态：**DONE_WITH_CONCERNS**（见文末"疑点"）。

## 提交
- fork `seabit-ai/mlx-engine`，分支 `lmk-spec-proc`（从 `lmk` 3b493b5 起），单个 commit **e053a72**
  "Speculative rounds for rows with logits processors: walk them position by position, llama.cpp style"。未 push；`lmk` 分支未动（ff 合入归 owner）。
  `.engine/mlx-engine` 现 checkout 在 `lmk-spec-proc`（= ENGINE_COMMIT，不游离）；`.engine/mlx-engine-spec` 回到 `lmk-spec`。
- lmk 分支 `spec-with-tools`：
  - 113fd2e `ENGINE_COMMIT` → e053a72；`tests/test_itest_chat.py` 新 itest `test_a_request_with_tools_drafts_when_a_draft_is_loaded`（需 `LMK_ITEST_DRAFT`，否则 skip），`chat()` 多返回 `lmk` 字段。
  - 245b56d research exp01–03（README 先写预期后跑，raw 在树里）。
  - push 顺序：先 push fork 的 e053a72，再 push lmk（install.sh 按 hash 取 tarball）。

## 引擎改动（`mlx_engine/model_kit/batched_vision/speculative.py`）
1. **SPD-028 修复**（`_plain_tick`）：轮之后 bonus 已在 `row.tokens` 里；普通步前对非 pending 行先 pop，`_advance` 之后再按"长度没变才 append"放回——
   processor 拿到的 context 与 decode-ahead 同口径（喂进之前的 token），bonus 只作为 last token 被看一次。
2. **`_processed_walk`**：对带 processor 的行逐位置：用 `_apply_logits_processors`（与普通步完全同一分发：守卫走 `process_last_token_with_context`、
   重复惩罚走 `process_last_token`、其余走通用分支）处理第 j 位 logits（context = bonus 之前已喂的 token + verify_input[:j]，last = verify_input[j]）→
   归一化 → 行的采样器抽（支持 mlx-vlm 的 positioned `sample_target`）→ 与草稿比，不等或到预算或到块尾即停。processor 只见被发出的 token，无回滚。
   - DFlash：`logits_at(j) = verify.logits[:, j, :]`。
   - MTP：`lm.speculative_logits_from_hidden(hidden[:, j:j+1])`；有 processor 的行不走融合 argmax（`all_greedy` 在有 processor 时为假）。
3. **`_can_round`**：去掉 processor 条件；只剩 top_logprobs（轮不产 logprobs）与 opt-out，代码里注了一行原因。
4. 采样路径：与现有轮同一原理（抽样比对，输出 token 永远是目标分布的一次抽样），且每发出一个 token 恰好一次采样调用、顺序相同——单测里同种子下与逐 token 路径逐位相同。

## 测试
- 新文件 `tests/test_batched_vision_speculative_processors.py`（8 个）：真 byte-level BPE tokenizer + 真 Qwen3.5 守卫（真 llguidance 语法）+ 假目标（原始 argmax 按脚本走，
  在四段里各设"诱惑"——思考段的 `<tool_call>`、调用体里的 EOS 与提前的 `</tool_call>`、调用后的正文）+ 假草稿器（提原始 argmax，所以诱惑必须被拒）。
  - SPD-028 两个（先写、先红）：通用 processor 看到 bonus 两次 / 调用体内 llguidance `ValueError: token "=" doesn't satisfy the grammar`。修后绿。
  - 贪心 dflash / mtp：轮的输出 == 逐 token + 守卫 == 脚本；四段（思考/正文/调用体/收尾）都有轮起草；有接受也有拒绝。
  - 采样 dflash / mtp：同种子下轮 == 逐 token。
  - 无守卫时同一世界会跑偏（证明诱惑有效）；`_can_round` 放行 processor、仍拒 top_logprobs。
  - 变异检查：关掉 walk（退回原 walk）→ 4 个 walk 测试红；history 错一位 → 采样测试红。
  - 已有 `test_batched_vision_speculative.py` 的 `_Round` 桩多收 `**kwargs`（仅此一行）。
- 引擎全量单测：55 failed / 5 errors / 362 passed（基线 354 passed）；**失败集与基线逐条相同**（60 条，diff 为空）。
- lmk `make test`：214 passed, 8 skipped；`make lint` 过。
- `LMK_ITEST=1 LMK_ITEST_KV_BITS=8 LMK_ITEST_DRAFT=<dflash2>` 跑 `test_itest_chat.py`：7 passed；带 tools 的请求 drafted 138 / accepted 126，completion 169，调用 file_write。
- `make itest`（无草稿，kv16）：7 passed, 2 skipped（cache-compat 需 LMK_COMPAT_DIR；新 tools-draft 测试无草稿时 skip）。

## itest 实验（27B-4bit，思考开 + low，kv8，dflash2，贪心 400）
| | exp01 fizzbuzz kv8 | exp02 fizzbuzz kv16 | exp03 copy kv8（推荐组合） | exp03 copy kv16 |
|---|---|---|---|---|
| 关·冷 vs 关·热 | 不同 | 不同 | 不同 | 不同 |
| 开 ×2 vs 关·热 | 不同（思考段 67 字符） | 不同（思考段 210 字符） | **逐字节一致** | 不同（思考段 65 字符） |
| decode 关·热 → 开 | 37.1 → 59.8/65.1 | 37.8 → 66.0 | 36.5 → 58.2（1.60×） | 37.6 → 59.0 |
| drafted / accepted | 258 / 220 | 同左 | 128 / 101 | 同左 |
| 思考 / 正文 / 调用体 | 66/50 · 4/2 · 188/168（89%） | 同左 | 26/14 · 2/2 · 100/85（85%） | 同左 |
| 采样一次 drafted/accepted | 190/162 | 274/235 | 140/105 | 134/110 |

- 分叉处（exp02）logits：`'\n\n'` 24.125 对 `'\n'` 24.000 = bf16 一个 ulp 的平手；且关草稿的冷/热跑之间就分叉，参照本身在这些 prompt 上不稳定。
- 调用后收尾段只有 1–2 个 token，真机实验里没有一轮的计数落在那一段（单测覆盖了）。

## 疑点（给 review）
1. **sha 对照只在 1/4 的组合上逐字节一致**。其余 3 个分叉都在思考段、都是近平手；exp02 量到一个是 1 ulp。我判为浮点平手而非 walk 的错，依据：单测逐位等价
   （含采样）、关草稿自己冷热就分叉、分叉位置守卫只掩 `<tool_call>`。但"开草稿 vs 关草稿一致"这条验收在 fizzbuzz prompt 上**没有达成**。
2. **开草稿的输出与计数在 kv8 与 kv16 上逐字节相同**（两个 prompt 都是），关草稿则不同。可能是轮里的校验/回滚路径没有体现量化 cache 的差异（推荐组合就是 kv8，
   值得查），也可能是关草稿路径本身在这些 prompt 上不稳定。没查——超出本任务，且不是本次改动引入的（没在旧 commit 上复现对照）。
3. 预算截断时 walk 的 accepted 按"最后一个发出的 token 不进 cache"算（与 decode-ahead 一致），和 mlx-vlm 的 `_speculative_walk` 数法差一，只影响已结束行的 cache。
4. 每位置一次 `.item()` 同步 + processor；实测带 tools 的请求 decode 仍 1.6×，没单独量每位置开销。
5. 没做 install、没重启常驻服务、没改模型页/README（按指示）。真机 kitten 流量验收与"模型页 Known issues 删除、速度表补一行"留给 controller。

---

# 第二轮（review 的 I1 与 minors），2026-09-25

状态：**DONE_WITH_CONCERNS**。

## 提交
- fork `lmk-spec-proc`（未 push），两个 commit：
  - **06f73c1**：原功能 commit（e053a72 经 autosquash 并入"walk 遇停止 token 即停"+ 对应单测）。e053a72 不再在任何分支上。
  - **42a248c**：I1 + DFlash 回滚口径——单独成一个 commit，因为它是自 b58a72e 起就有的独立 bug（MTP 一样受影响），不是 walk 的一部分。
- lmk `spec-with-tools`：3d52ae7（`ENGINE_COMMIT` → 42a248c），e86bced（notes SPD-032..034 + exp04）。
- `.engine/mlx-engine` 在 `lmk-spec-proc`（= ENGINE_COMMIT）；`.engine/mlx-engine-spec` 回到 `lmk-spec`。

## 修法与理由
- **I1**：选"报 all_tokens 去掉未喂的尾 token"（`_response`，`_emit_pending` 与 `_round` 共用），不选"交出前再把它喂进 cache"。理由：命中数与 cache 实际覆盖
  严格一致；不多一次前向（结束时多 ~25 ms）、不碰草稿器与 GDN 记录状态；代价只是下一轮多 prefill 1 个 token（下一轮本来就要 prefill 至少 1 个）。
  普通路径（decode-ahead）发出的 token 在发出时就已喂进，`all_tokens` 本来就对，不动。
- **DFlash 回滚**：先截断，再按 `len(cut)-1` 回滚（与 MTP 轮同口径）；草稿统计仍记 walk 的接受数。
- **walk 停在停止 token**：`_processed_walk(stop=…)`，由 `_round` 传 `self.stop_criteria`。

## 测试
- 新单测（先红后绿）：两种草稿器 × 有/无 processor，行在轮里以 EOS 结束 ⇒ `all_tokens == 目标实际喂过的 token`（4 个里 3 个红；dflash 无 processor 那个因两个 bug 相互抵消碰巧绿）；
  `max_tokens` = 1/7/10/13（落在块中间）× 两种草稿器 ⇒ 输出与逐 token 相同且 cache 对齐（8 个红，输出本身都相同，红在 cache）；walk 停在 EOS（红）。
  变异：回滚改回按接受数 → 1 红；去掉 walk 的 stop → 1 红。旧测试 `test_a_stop_token_inside_the_round_ends_the_row_with_its_cache` 的期望从 `[1,5,7,2]` 改为 `[1,5,7]`（旧期望正是 bug）。
- 引擎全量：55 failed / 5 errors / 375 passed；失败集与基线逐条相同（diff 为空）。
- `make test` 214 passed, 9 skipped（多出的 1 个 skip 是新 itest 在非 itest 模式下）；`make lint` 过。
- chat itests（kv8 + dflash2）7 passed，带 tools 的请求 drafted 138 / accepted 126；`make itest` 7 passed, 2 skipped。

## I1 在真机上复现了吗
复现了（exp04，旧引擎 e053a72，kv8 + dflash2，思考开 + low）：带 tools 的第一轮开草稿以工具调用结束，热 entry 807 token、KV offset 806（缺 `<|im_end|>`）；
Qwen 模板重渲染第一轮与生成逐 token 相同，第二轮整段命中（trim 0），引擎报 cached 807。修复后 806/806、报 806。净效应在这一例里小：第二轮首位置 top-5 logprob
（旧对修复后，同为热命中）差 ≤ 0.25，60 token 文本相同——没有观察到乱码。没有加 pytest itest：要断言需要读引擎内部（热 entry、KV offset），lmk 侧不该碰；
证据留在 exp04 的脚本与 raw 里。

## 疑点
1. **SPD-034**：kv8 上"热 cache 续算"与"磁盘恢复 512 + 重新 prefill 320"对同一 prompt 的首位置 logprob 差到 2.75，修复前后都在。与 exp01–03 关草稿冷/热 sha 不同可能同源。未查。
2. lmk 的 113fd2e 里 `ENGINE_COMMIT` = e053a72，那个 commit 已被 rebase 掉、不会被 push；若按 commit 逐个跑 CI 会取不到 tarball。合并前可考虑把 113fd2e 与 3d52ae7 在 lmk 侧 squash。
3. 第一轮的疑点 1（fizzbuzz 上开/关 sha 不一致是平手）不变；review 已解第一轮疑点 2。
