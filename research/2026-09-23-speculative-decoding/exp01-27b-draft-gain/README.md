# exp01：Qwen3.8-27B-4bit 配草稿模型，顺序路径上的 decode 增益

日期 2026-09-23。m3u（M3 Ultra 96GB），引擎 08f0c07，`mlx-lm 0.31.3`。owner 拍板 fork mlx-engine 后，
设计文档 §3 要先量一次投机解码的真实收益，决定 fork 上先补投机解码还是先补 KV 量化
（判据：≥1.8× 先投机解码，≤1.3× 先 KV 量化）。

## 怎么量
- 直接构造引擎的顺序路径 `ModelKit`（不是 lmk 走的批处理路径——那条不支持草稿），主模型
  `lmstudio-community/Qwen3.8-27B-MLX-4bit`（lmk pull 的那份），草稿 `mlx-community/Qwen3.5-4B-MLX-4bit` 与 `Qwen3.5-2B-MLX-4bit`。
- 常驻 lmk（122B-48GB）先 `lmk down`，跑完 `lmk up`，GPU 只给实验。
- 贪心（temp 0），思考关（`enable_thinking=False`），每题 400 token 输出，三种题各跑 2 遍：
  1. `story`：bench 里的 decode 题（"Write a 250-word story about a lighthouse keeper."），与 `docs/benchmarks.md` 的 39.5 tok/s 可比；
  2. `code`：写一个带 docstring 和测试的 Python 函数（代码可预测性高，猜中率应高于散文）；
  3. `copyedit`：把给定的一段文字改错别字后原样返回（大量照抄输入，猜中率最高，模拟 agent 改代码时的"复述"）。
- 五臂：`main`（无草稿）· `4b-n2` · `4b-n4` · `2b-n2` · `2b-n4`（n = 每轮草稿 token 数，mlx-lm 默认 2）。
- 记：decode tok/s（首 token 之后的 token 数 / 首 token 到末 token 的时间，和 bench 同口径）、
  草稿命中率（`from_draft` 标记的 token 占比）、prefill 时间（草稿模型也要 prefill 一遍 prompt，这是额外代价）。

## 预期（跑之前写）
- **顺序路径无草稿的 decode ≈ 批处理路径**：38–42 tok/s（批处理路径 bench 39.5）。把握：高。
- **4B 草稿、n=2、story**：草稿 decode 受带宽限制约 5× 快于主模型（权重 2.85 对 16.1 GiB），一轮 = 2 步草稿 + 1 步主模型验证，
  验证 3 个 token 的成本≈单步 decode（带宽受限）。假设散文贪心猜中率 70%：每轮期望 1+0.7+0.49≈2.2 token，
  轮时 25 ms + 2×5 ms = 35 ms ⇒ ≈60 tok/s，**1.4–1.7×**。把握：低——猜中率就是这次要量的东西。
- **n=4 不比 n=2 快多少**：多猜的 token 命中概率指数下降，轮时线性增加，散文上 n=4 ≈ n=2 ±10%；copyedit 上 n=4 明显赢（命中率 90%+ 时长草稿划算）。把握：中。
- **2B 草稿**：草稿更快但猜中率更低，散文上和 4B 打平或略输；copyedit 上略赢。把握：低。
- **按题型**：copyedit > code > story；copyedit 有望 **2.5× 以上**。把握：中。
- **落到判据**：story 这种最不利的题预计落在 1.3–1.8 的灰区，code/copyedit 过 1.8。若如此，结论是"投机解码在 agent 负载（代码、复述）上收益够大，先做投机解码"。

## 怎么跑
`run.sh`（`lmk down` → `run.py` → `lmk up`）。原始每次运行的 JSON 在 `raw/`，`results.md` 是汇总。

## 结果（2026-09-23）：草稿臂一个都没跑起来——mlx-lm 的投机解码不支持 Qwen3.8 的混合注意力

`raw/main-*.json`、`run.log`。`main` 臂（顺序路径、无草稿）三题六次 decode 全部 **39.2–39.5 tok/s**，与批处理路径 bench 的 39.5 一致——
预期"顺序路径 ≈ 批处理路径"命中。

加载 4B 草稿后第一条请求即抛：

    ValueError: Speculative decoding requires a trimmable prompt cache (got {'ArraysCache'}).

原因（SPD-003）：mlx-lm 的投机解码靠"验证失败后把 KV cache 裁回去"（`trim`），要求每层 cache 可裁剪；Qwen3.8-27B 64 层里 48 层是
linear attention，mlx-lm 给它们用 `ArraysCache`（固定大小状态，不可裁剪）。所以**在 mlx-lm 上，外挂草稿模型对 Qwen3.5/3.8 整个家族都不可用**，
和引擎版本无关。四个草稿臂没有数据，"4B/2B 草稿、n=2/4"的预期无法验证。

这条路的死因把我们引到了另一条：mlx-vlm（批处理路径的底座）自己实现了投机解码，且专门处理了混合注意力的回滚
（`rollback_speculative_cache` 里对 SSM/线性层状态单独处理），并支持 Qwen3.5 家族**原生 MTP 头**做草稿。见 exp02。
