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

## 方法（`compute.py`，读 exp01 / exp02 的 raw，不加载模型）
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
