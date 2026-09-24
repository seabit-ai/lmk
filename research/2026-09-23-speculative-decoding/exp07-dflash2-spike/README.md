# exp07：DFlash 2 草稿器在 M3 Ultra 上——用上游 mlx-vlm 0.6.16 自己的循环先量一遍（spike）

日期 2026-09-24。m3u（96 GB）。目的：在花引擎升级和接线的钱之前，回答"DFlash 2 在这台机器、这个模型上到底给多少、准不准"。
不碰引擎：scratch 里独立的 venv（mlx-vlm 0.6.16，PyPI），主模型用 `lmk pull` 那份 `lmstudio-community/Qwen3.8-27B-MLX-4bit`，
草稿器 `incoai/Qwen3.8-27B-DFlash2`（z-lab 镜像，2B 参数 bf16 约 4 GB，block 8，top-k + selector）。
对照：我们的 MTP 头（exp05/06 与 `docs/benchmarks.md`）：贪心 code 1.49× / prose 1.14×，每轮接受 2.7，采样接受率 86–88%。

## 事实（跑之前查的）
- mlx-vlm v0.6.16 加了 `drafters/dflash2`（`CandidateSelector`、`GroupedDynamicCausalConv`，起始块 3）和 "exact Qwen3.8 27B DSpark"；我们引擎钉的 0.6.12 没有。
- mlx-dspark（另一个 MLX 实现）在 M4 Pro 48 GB 上：27B **4-bit 2.30×**（每轮 5.14），8-bit 3.63×；说"到浮点平手为止逐字节一致"。
- 升级到 0.6.16 对我们 fork 的代价：`qwen3_5/language.py` 改了 873 行，我们挂钩的 7 个符号 6 个还在，`_target_verify_left_padded_attention` 没了。

## 怎么量（`run.py`）
exp01 的三题（story / code / copyedit），贪心 400 token：普通解码 → DFlash2（sha 对照、tok/s、每轮接受）；temp 1.0 的接受率；
mlx-vlm 的 `generate`（单序列路径）。跑时 `lmk down`（GPU 争用），跑完 `lmk up`。原始输出 `raw/`。

## 预期（跑之前写）
- E1 加载：0.6.16 的 `load_drafter` 认出 `dflash2`，草稿器和 4-bit 主模型兼容（词表、hidden）。把握：中高。
- E2 贪心 code / copyedit 逐字节与普通解码一致；story 可能分叉（MTP 也分叉）。把握：中。
- E3 速度：code ≥ 2.0×（M4 Pro 2.30×，M3 Ultra 带宽更高但草稿一步的固定开销比例也不同），prose ≥ 1.5×。把握：中低。
- E4 每轮接受 ≥ 4（他们 5.1–5.5）。把握：中。
- E5 采样 temp 1.0 接受率 ≥ 70%。把握：低——v2 的 selector 对采样怎么处理没查。
- 落到判据：code ≥ 2.0× 且逐字节一致 ⇒ 值得升级引擎 + 接线（DFlash2 作为第二种草稿器）；否则记下数，继续用 MTP 头。

## 过程与结果（`raw/run.log`、`raw/*.json`）
1. 环境：`uv venv` + `pip install mlx-vlm==0.6.16`（PyPI 没把 mlx-lm 列为依赖，要另装，装到了 mlx-lm 0.31.3）。草稿器 4 GB bf16，2 分钟下完；
   `load_drafter` 认成 kind `dflash`（DFlash2 是 DFlash 的子类）；config：block 8、top-k 16、`target_layer_ids [5, 19, 33, 47, 61]`（要目标模型 5 层的隐状态）。
   exp01 的 `run.py` 顶层 import 引擎，这个 venv 没有——改成从源码里 exec 顶层赋值取 `PROMPTS`。**E1 命中。**
2. 主结果（mlx-vlm 自己的循环，贪心 400 token，两遍一致取平均）：

| | 普通解码 | DFlash2 | 倍数 | 每轮接受 | 接受率 | MTP 头（exp02 同循环 / exp03 引擎） |
|---|---|---|---|---|---|---|
| story | 38.2 | 39.2 | **1.03×** | 2.04 | 204/392 = 52% | 1.22× / 1.14× |
| code | 38.1 | 46.6–48.7 | **1.22–1.27×** | 4.26 | 306/374 = 82% | 1.59× / 1.49× |
| copyedit | 37.9 | 55.7–56.5 | **1.47–1.49×** | 4.93 | 106/106 | 1.70× / 1.48× |

   **三题全部逐字节一致**，连 story 也一致（MTP 头在 story 上分叉）——E2 超预期。采样 temp 1.0：code 接受 80%、story 47%。
   **E3 落空**（code 1.25× 对判据 2.0×）；E4 命中（每轮 4.3–4.9 对 MTP 的 2.7）；E5 命中（80%）。
3. **为什么每轮接受更多、反而更慢**：草稿一步太贵。2B bf16 的草稿器每轮读 4 GB（目标一步 15 GB 的四分之一），再加 5 层隐状态投影、
   动态卷积和 selector；MTP 头只有一层、810 MB。M3 Ultra 的目标一步本来就快（38 tok/s），草稿的固定开销占比大；mlx-dspark 在 M4 Pro 上
   报 2.30×，那台机器普通解码只有 14.7 tok/s，同样的草稿开销摊在慢得多的目标步上。**DFlash2 的收益随机器带宽升高而缩小**——这是猜的原因，
   没单独量草稿一步的耗时；能证伪的实验：草稿器量化到 4 位后倍数是否明显上升（`raw/run_extra.log`）。
4. **额外两臂**（`run_extra.py`，`raw/run_extra.log`；草稿器用 `mlx_vlm.convert -q --q-bits 4` 量到 4.5 bit/权重、约 1.1 GB）：

| 贪心 400 token | story | code | copyedit |
|---|---|---|---|
| DFlash2 缺省（动态块） | 1.03× | 1.22–1.27× | 1.47–1.49× |
| 固定块 8 | 1.00×（2.0/轮） | 1.17×（4.9/轮） | 1.54×（7.4/轮） |
| 4 位草稿器 | 1.01× | **1.14×**（4.3/轮） | 1.32× |
| 4 位草稿器 + 固定块 8 | 1.05× | **1.26×**（4.9/轮） | **1.66×**（7.4/轮） |

   全部逐字节一致。**第 3 条的猜测被证伪**：草稿器从 4 GB 缩到 1.1 GB，速度没变甚至略降——开销不在读权重，在草稿一步的计算
   （动态卷积、selector 的 top-16 路径搜索、5 层隐状态的投影）和循环本身。固定块 8 让每轮接受涨到 4.9–7.4，但每轮多做的草稿计算把收益吃掉。

## 结论（对判据）
- code 最好 1.26×，**没过 2.0× 的判据；也没超过我们现有的 MTP 头（1.49×）**。DFlash2 在这台机器上的价值是"逐字节一致 + 采样接受 80%"，不是速度。
- 不为它升级引擎（`language.py` 873 行的返工 + 重做量化校验补丁）、不接第二种草稿器。27B 继续用 MTP 头。
- 留一句给将来：DFlash/DSpark 这条线该给 **Gemma 4** 用（没有 MTP 头，`gemma4_dflash` / `gemma4_dspark` 上游都有），那时要一起升级 mlx-vlm；
  以及**收益随机器带宽升高而缩小**这一点，在带宽低的 Mac（M4 Pro 一类）上应重新量——mlx-dspark 的 2.30× 可能在那些机器上成立。
- 副产品：草稿器 `incoai/Qwen3.8-27B-DFlash2` 留在 HF cache（4 GB），4 位版在 scratch 里，随会话清理。
