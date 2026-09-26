# exp01：带 tools 的 agent 请求开/关草稿——输出逐字节一致吗、起草了多少（按段拆开）

日期 2026-09-25。m3u（M3 Ultra 96 GB）。fork 分支 `lmk-spec-proc` e053a72（裁决 A：投机轮对带 processor 的行逐位过 processor + SPD-028 交接修复），
lmk 分支 `spec-with-tools`，`ENGINE_COMMIT` 指向 e053a72。常驻服务不停（另占约 16 GB + 草稿器）。

## 怎么量（`run.py`）
27B 页的推荐组合：`qwen3.8-27b-4bit`、`kv_cache_bits: 8`、思考开 + `reasoning_effort: low`、缺省草稿器（dflash2，4-bit）。
走 lmk 的 `MlxEngine`（与服务同一条路：同一个 chat 模板渲染、同一组采样参数名），一次加载，草稿开关用引擎的逐请求开关 `speculative_decoding_toggle`。
prompt：kitten 式 system + 三个工具（file_read / file_write / run_command）+ 用户"写一个 fizzbuzz.py"——思考段、正文、调用体（文件内容，复述型，
草稿最该赚的一段）、调用后收尾四段都可能出现。
1. 贪心 400 token：关草稿 ×2（第一次冷、第二次 prompt cache 命中，先确认"关 vs 关"本身一致），开草稿 ×2。每次记全文 sha、token 数、decode tok/s。
2. 开草稿的每次都按段拆 drafted / accepted：每读一段文本就读一次草稿器的累计计数，差值记到"读完这段后所处的段"（思考 / 正文 / 调用体 / 调用后）。
   这是近似：一轮的计数在这一轮吐出的第一个 token 时读到，而一轮的草稿跟在 bonus 之后，段边界处会错一位。
3. 模型缺省采样（temp 1.0 / top_p 0.95 / top_k 20，kitten 的真实用法）开草稿一次：只报按段 drafted / accepted，不比 sha。
原始输出：`raw/run.txt`（stdout）、`raw/results.json`（全文与计数）。

## 预期（跑之前写）
- E1 不崩：调用体内起草不再触发 llguidance 的 ValueError（SPD-028 已修、walk 只喂被接受的 token）。把握：高（单测覆盖了四段 + 交接）。
- E2 贪心开/关草稿 sha 逐字节一致。把握：**中低**——kv8 上 exp12 的 code 题就因量化注意力 L>1 与 L=1 不位级一致而分叉（sha 与 exact verifier 下相同，
  不是逻辑错）。processor 只是掩码，不引入新的浮点差，但也消不掉这一种。若分叉：记分叉位置与所在段，再在 kv16 上补跑同一 prompt（新实验 exp02），
  kv16 一致即归为 exp12 已知的量化浮点平手，kv16 也分叉才是 walk 的错。
- E3 带 tools 的请求 drafted > 0，四段里至少思考、调用体两段都有起草。把握：高。
- E4 接受率：调用体最高（复述型文本，exp12 copyedit 类 70%+），思考/正文较低（story 类约 45%）；整体 decode 比关草稿快 1.2–1.6×。把握：中低（未在轮里量过每位置
  processor 的开销，SPD-025 估 0.4–3.7 ms/位置；块宽 4–8 时一轮多几到十几毫秒）。
- E5 采样下同样 drafted > 0，接受率比贪心低（exp12：temp 1.0 code 78–81%）。把握：中。

## 过程与结果（`raw/run.txt`、`raw/results.json`）

| kv8，思考开 + low，dflash2，贪心 400 | token | decode tok/s | sha |
|---|---|---|---|
| 关草稿，冷 | 284 | 30.0 | 0d5ff0db7afa |
| 关草稿，热（命中 512） | 254 | 37.1 | ea1f45d8f772（**与冷跑就不同**） |
| 开草稿 ×2（热） | 286 | 59.8 / 65.1 | f9e1be8e5119 ×2（与关·热不同） |
| 开草稿，模型缺省采样 | 233 | 51.7 | — |

按段（贪心开草稿，两次完全相同）：思考 drafted 66 / accepted 50（76%，17 轮）；正文 4 / 2（1 轮）；调用体 188 / 168（89%，47 轮）；合计 258 / 220，65 轮。
采样：思考 10 / 6、正文 2 / 2、调用体 178 / 154（87%）；合计 190 / 162。调用后收尾段只有一两个 token，没有一轮落在那里的计数。

- E1 命中：不崩，调用体里起草、接受都正常。
- E2 落空，但不能归到 walk：分叉在思考段第 67 字符（关：`fizzbuzz(n)`；开：`` `fizzbuzz(n)` ``）；而**关草稿的冷跑与热跑之间就已经分叉**——
  这个 prompt 在 kv8 上对 prefill 路径本身就敏感。见 exp02（kv16）与 exp03（换 prompt）。
- E3 命中：drafted 258 > 0；思考、调用体都有起草。
- E4 命中方向、超出幅度：调用体 89% 最高、思考 76%；decode 1.6–1.75×（37.1 → 59.8/65.1）。
- E5 命中：采样下 drafted 190、调用体 87%。
