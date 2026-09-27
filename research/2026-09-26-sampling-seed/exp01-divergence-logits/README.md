# exp01：同 seed 的两处分叉，在 logits 上差多少

起因（owner 跑的 itest，`../../2026-09-26-context-check/raw/itest2.log`，27B-4bit、kv16）：
- (i) 冷算 vs 磁盘恢复 256 后续跑，同 seed：带 tools 的请求分叉（不带草稿在第 15 个字符："We need answer user" / "We need answer to user"；带 dflash2 在第 173 个字符）。
- (ii) 短 prompt，同 seed，投机关 vs 开（dflash2）：不相同（起草 106、接受 58）。

## 做法
不比文本，比 logits：给按位置抽样的 `SeededSampler.sample_target` 包一层，记下每个生成位置交给采样器的 logprobs（全词表，取 top-20）
和抽中的 token；同一位置被抽多次时（投机轮里被拒之后的位置下一轮重抽）留最后一次——那才是发出去的那次。
两次运行逐位置对齐，报告：第一个分叉位置 j；j 之前所有位置 top-20 并集上的 max |Δlogprob|；j 处两边的 top-5、抽中 token 的概率，
以及按抽样的真实机制判断是不是"临界"——同一把 key 下，两边 top-k/top-p 截断后的分布只要在那一抽的边界附近差一点就会抽到不同 token。
- (i) `run.py cold-vs-restore`：不带草稿，plain / tools 两种 prompt × 3 个 seed。
- (ii) `run.py spec`：带 dflash2，同一个短 prompt × 3 个 seed，先热身一次，投机关与开都从同一个恢复点起。

## 预期（跑前写）
- (i) 冷算与恢复后续跑：分叉前各位置 max |Δlogprob| < 0.1（浮点级：恢复点前后分块形状不同，归约顺序不同）；分叉处两个候选概率相近。
  把握：中。若 > 0.5 就是 bug（owner：cache 只允许舍入级的差别）。
- (ii) 投机开关：校验走一次块 forward（多 token），普通路径一次一个 token，矩阵形状不同 → 同样是浮点级差别，< 0.1。把握：中。
  若投机那边在分叉前就大幅偏离，查 DFlash 校验的 logits（`_dflash_target_logprobs`）或 GDN 回滚。

## 追加（跑前写）：带 dflash2 时冷算 vs 恢复
itest（dflash2 + tools）里冷算 vs 恢复在第 2 个位置就分叉，那里 TV 0.29、max|Δ| 0.75——比不带草稿时大一个量级。
预期：分叉前几个位置里，一边是普通步抽的、一边是投机轮的校验块抽的（恢复路径与冷路径进第一轮的时机不同），
差别仍是"块 forward vs 单 token forward"的数值差，但在个别位置可到 0.3。把握：低。若同一种抽法下也差这么多，就是 bug。
`run.py spec-cold-vs-restore`。

## 结果（m3u，27B-4bit，kv16；`raw/`）
**1. 浮点噪声有多大（对照，`raw/chunking-control.txt`）**：同一个 prompt 一次读完 vs 分两块（256 + 其余），没有 cache 参与：
最后位置 top-20 的 max |Δlogprob| 0.22–0.60；top-1 差 0.006–0.10。logits 是 bf16，这个量级上一格就是 0.125——"< 0.1"对 top-20 做不到。
同一调用重复两次：0。

**2. (i) 冷算 vs 恢复，不带草稿**（`raw/cold-vs-restore*.txt`、`raw/pos0*.txt`）：第一个 token 的分布 TV ≤ 0.009，p>0.05 的 token 上 Δ 0.000；
逐位到分叉前 TV ≤ 0.063，Δ(p>0.05) ≤ 0.5。分叉处是抽样边界（例：`vet` 0.687 对 0.535，同一把 key 落在两边分布的边界之间），
不是 bug。**预期对了一半**：量级是浮点级，但比我写的 0.1 大（bf16 的一格）。

**3. (i) 冷算 vs 恢复，带 dflash2——这是 bug**（`raw/spec-cold-vs-restore.txt`、`raw/pos0-reference.txt`）：第一个 token 的分布
TV **0.24–0.30**，p>0.05 的 token 上 Δ 最大 0.80（`We` 在冷算里 −1.375，恢复后 −0.75）；对照同进程里一次普通 forward，
偏的是恢复那一边（0.29–0.33），冷算只差 0.04–0.06。
- 原因：DFlash 草稿器让每次 forward 都带 `capture_layer_ids`，文本请求因此走不了 fork 的快路径，进 mlx-vlm 自己的 call；
  那里没有存着的 RoPE 状态时位置从 0 数起，而引擎在每次 prefill 前后都会清掉这个状态（`_clear_qwen3_5_text_rope_state`）。
  冷算带显式 position_ids 所以没事；**恢复之后的那一段按位置 0..n 编码**，写进 cache 的 key 也是（并随 256 块存进磁盘）。
  （探针：`probe_kwargs.py` → `raw/kwargs-probe.txt`（修后重跑：调用方传的 kwargs 不变，位置在补丁里补上）——恢复后的 116 token 调用没有 position_ids、没有 rope_deltas、fa cache offset 256。）
- 修（fork `39c17a2`）：这类调用给出 fast path 会用的位置（cache offset + arange；批里逐行）。修后：冷算与恢复**逐位完全相同**（TV 0.000），
  与普通 forward 差 0.01–0.07（`raw/pos0-reference-after-fix.txt`）；itest 全过（`raw/itest-full-after-fix.txt`）。
- **与事故 000193 的关系（推测，未验证）**：owner 机器的配置正是 dflash2；那次从磁盘恢复 83712 个 token、prefill 1186 个——
  按这个 bug，那 1186 个 token 被编码在位置 0..1185，新一轮在全注意力层看来"位于对话最前面"。模型说"这只是个 system prompt，
  用户什么都没问"与此相符。重放三次都正常不矛盾：重放时恢复点可能不同（热 cache / 不同 256 边界）。验证办法：在修前的引擎上用
  000193 的请求 + dflash2 从磁盘恢复复现，再换修后的引擎比。
- **磁盘上已有的 cache 被污染了**：修前带 dflash2 时，每个"恢复后再 prefill"写下的 256 块里，恢复点之后的 key 位置是错的。
  cache 身份不含引擎 commit，升级引擎后这些块还会被恢复。**建议合并时 `CACHE_FORMAT_VERSION` +1**（清掉旧 cache），owner 定。

**4. (ii) 投机开 vs 关（修后，`raw/spec-profile-after-fix.txt`）**：每个位置的 TV ≤ 0.10，p>0.05 的 token 上 Δ ≤ 0.375，按块内偏移看没有哪一格特别大；
3 个 seed 里 2 个在 70 个 token 附近分叉，分叉处是抽样边界（`from` 0.287 对 0.253）。校验走块 forward、普通步走单 token forward，
和"分块读 prompt"是同一类浮点差。**预期对（浮点级），但量级是 bf16 的量级，不是 < 0.1。**
短 prompt 上"逐字相同"的预期落空——猜错在把"贪心下短 prompt 逐字相同"搬到了采样上：贪心只在 argmax 附近敏感，采样在每一处边界都敏感。

**itest 的改法**：冷算 vs 恢复、投机开 vs 关都改成逐位置比分布（到第一次分叉为止）：TV ≤ 0.15、p>0.05 的 token 上 |Δlogprob| ≤ 0.75
（约为实测最大值的 1.5–2 倍）；SPD-034 那种 2.75 或本次 bug 的 0.30 / 0.80 都会被抓住。同路径的重放仍断言逐字相同。
