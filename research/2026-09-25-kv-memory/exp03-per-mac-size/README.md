# exp03：按 Mac 内存自动选 KV 精度——每档内存下两种精度的窗口与"tokens in memory"上限（计算，不是实测）

日期 2026-09-25。输入：lmk 的公式（`lmk/models.py` 的 `context_on`、`MAC_MEMORY_SIZES_GB`、`GPU_SHARE_OF_MEMORY`；`lmk/engine.py` 的 `token_budget`
= (working set − 3 GiB − baseline) ÷ KV B/token），`qwen3.8-27b-4bit` 的 `MemoryFit`（baseline 14.95 GiB，KV 65,536 / 34,816 B/token），
以及 exp01 / exp02 在 m3u（96 GB）上量到的进程占用。**只有 96 GB 这一台是量的，其余每一格都是算的。**

## 要回答
16 / 24 / 32 / 48 / 64 / 96 / 128 / 192 GB 的 Mac 上，kv8 与 kv16 各自：
1. 引擎会给的窗口（lmk 的 `context_on`，即引擎的公式）与 lmk 的 tokens-in-memory 上限（公式）；
2. 把实测的"公式没算进去的那部分"（MLX 缓冲池，及实测 peak 高出公式的部分）加回去之后，窗口用满时进程会不会超出 GPU working set；
   不设缓冲池上限与设上限（exp02 选出的值）各一列；
3. 由此得出的"自动选 KV 精度"规则长什么样。

## 方法（`run.sh` → `compute.py`，读 exp01 / exp02 / exp06 的 raw，不加载模型）
- 公式列：直接调 `lmk.models.context_on(m, gb, bits)`；上限按 `engine.token_budget` 的公式，working set = gb × 0.81 GiB（lmk 的假设，只在 96 GB 上核过 77.76 GiB）。
- 实测修正列：从 exp01 每档（8k–128k）读三样——① 冷 prefill 请求的 MLX peak 与 footprint，② 解码请求中的 footprint 最大值，
  ③ 公式对同一上下文预测的 prefill 峰值（baseline + tokens × (KV + prompt_inputs + attention × 2048)）。
  "公式外的量"= footprint 最大值 − 公式预测，按 token 做线性拟合（两种精度各一条；设上限的一条来自 exp02 的 kv16，kv8 设上限没量，按"缓冲池 ≤ 上限"推）。
  再对每档 Mac 求：公式给的窗口用满时预测的 footprint、它与 working set 的差；以及 footprint 不超过 working set 的最大上下文。
- 假设（标在结果里）：每 token 的系数与机器无关（同一模型同一引擎）；缓冲池的大小只随上下文走，与总内存无关；macOS 在别的 Mac 上也给 GPU 81%。
  这三条在别的机器上都没核过。

## 对比基准
- 公式在 96 GB 上：两种精度窗口都是 262,144；上限 kv8 1,844,474、kv16 979,877（exp01 实测日志）。
- exp01 kv8 128k（先跑完的一半）：冷 prefill 请求 MLX peak 38.8 GB、footprint 44.9 GB，缓冲池最高 18.4 GB；解码请求 footprint 34–39 GB。
  公式对 128k 冷 prefill 的预测（step 2048）≈ 16.05 + 131k × 143,360 B ≈ 34.9 GB——实测 peak 高 3.9 GB（约等于那 3 GiB reserve），footprint 高 10 GB。

## 预期（算之前写）
- **P1 公式列**：kv16 窗口 16 GB 装不下 / 24 GB 15.9k / 32 GB 85k / 48 GB 224k / 64 GB 起 262k；kv8 24 GB 22.8k / 32 GB 123k / 48 GB 起 262k。
  上限 kv16：24k / 131k / 343k / 555k / 980k / 1.40M / 2.25M（24→192 GB），kv8 约为 1.9 倍。把握：高（就是公式）。
- **P2 不设缓冲池上限时，公式把窗口排满的小内存 Mac（24 / 32 / 48 GB）用满窗口会超出 working set 数 GB**（缓冲池 + peak 超公式的部分）；
  64 GB 起两种精度都有富余（窗口封顶 262k，余量大）。设 1–4 GB 上限后超出量降到约 reserve 以内。把握：中（缓冲池"只随上下文走"是假设）。
- **P3 自动选择规则**：kv16 在它的窗口到顶（262k）且余量足够两个长请求并发时选 kv16，否则 kv8——即 **64 GB 及以上 kv16，48 GB 及以下 kv8**；
  48 GB 是边界（kv16 窗口 224k 对 kv8 262k，上限 343k 对 645k）。把握：中（边界取决于"并发两个长请求"算不算必须，这是产品裁决，不是计算）。

## 结果（`results.md`；2026-09-26 算，改过一次方法，见下）
方法跑前写的是"公式外的量 = 实测 − 公式预测，按 token 拟合"。算的时候改成了更直接的：每个解码请求的"最坏占用" = max（0.5 s 轮询的 footprint，
MLX 每请求 peak + 1.12 GB 非 MLX 部分），对 token 数拟合直线——不设上限用 exp01 的 8k–128k 四档，设上限（4 GiB，exp05 选的）用 exp02（kv16）/ exp06（kv8）的 32k 与 128k 两档。
冷 prefill 不进拟合（它的峰值随 prefill 步长走，引擎在小内存上会降步长，公式管这件事）；**从 cache 还原长前缀不在引擎的公式里**，这是本表要补的。

拟合（96 GB，单请求）：不设限 kv8 19.6 GB + 149 kB × token（4.3 × KV），kv16 19.8 + 270 kB（4.1 × KV）；
4 GiB 上限 kv8 20.0 + 126 kB（3.6 × KV），kv16 19.5 + 189 kB（2.9 × KV）。引擎公式按每 token KV + 34.8 kB（步长 512）算：kv8 69.6 kB，kv16 100.4 kB。

| Mac | 窗口（公式）kv8 / kv16 | tokens-in-memory 上限（公式）kv8 / kv16 | 一个请求能真用到的上下文，不设限 kv8 / kv16 | 4 GiB 上限 kv8 / kv16 |
|---|---|---|---|---|
| 16 GB | 装不下 | – | – | – |
| 24 GB | 23k / 16k | 46k / 24k | 9k / 4k | 7k / 7k |
| 32 GB | 123k / 85k | 246k / 131k | 55k / 30k | 62k / 44k |
| 48 GB | 262k / 224k | 645k / 343k | 149k / 81k | 173k / 117k |
| 64 GB | 262k / 262k | 1.05M / 555k | 242k / 133k | **262k** / 191k |
| 96 GB | 262k / 262k | 1.84M / 980k | 262k / 236k | 262k / **262k** |
| 128 GB | 262k / 262k | 2.64M / 1.40M | 262k / 262k | 262k / 262k |
| 192 GB | 262k / 262k | 4.24M / 2.25M | 262k / 262k | 262k / 262k |

（"能真用到的"= 拟合直线碰到 GPU working set（内存 × 0.81）的那个 token 数，封顶 262k；96 GB 以上与 64 GB kv8 的 128k 以外部分是外推。）

对预期：
- **P1 命中**：公式列与押的一致（`context_on` 与 `token_budget` 的公式本身）。
- **P2 命中但方向更重**：押的是"小内存 Mac 超出 working set 数 GB"，算出来 32 GB 超 8–15 GB、48 GB 超 11–38 GB。**而且设了上限也超**：超出的主体不是缓冲池，
  是还原前缀时的瞬时 active（约 3 × KV，exp01 时间线），公式只按 1 × KV + 34.8 kB 留。24 GB 两行受截距 ±0.5 GB 的误差支配，别当真。
- **P3 部分命中**：规则（设 4 GiB 上限的前提下）"kv16 的可用上下文到顶才选 kv16"→ **96 GB 及以上 kv16，64 GB 及以下 kv8**——比押的（64 GB 起 kv16）高一档：
  64 GB 上 kv16 只能真用到约 191k，kv8 能用满 262k。不设上限时 96 GB 的 kv16 也只到约 236k（本机现行配置，外推，262k 的请求没跑过）。
- 另外：**tokens-in-memory 上限按 1 × KV/token 计**，实测一个在内存里的 token 占 3–4 × KV（瞬时）。96 GB 上它从不起作用（最多 2 个请求 × 262k = 524k，低于 0.98M / 1.84M），
  在 48 GB 以下它会放进装不下的两个请求——但两个并发请求本组没量。
