# exp02：Qwen3.8-27B-4bit 用自带的 MTP 头做草稿（mlx-vlm 的投机解码）

日期 2026-09-23。m3u（M3 Ultra 96GB），mlx-vlm 0.6.12（引擎 08f0c07 钉住的 321514d）。exp01 证明 mlx-lm 那条路对
Qwen3.8 走不通（SPD-003）；owner 提醒"Qwen3.8 自带草稿模型"——对：原版 `Qwen/Qwen3.8-27B` 第 18 个分片里有 15 个 `mtp.*`
张量（一层 full attention + fc + norm，共享主模型的 embedding 和 lm_head），MLX 转换（lmstudio-community、mlx-community）都把它剥掉了。

## 怎么量
- 草稿器：只下原版第 18 个分片（3 GB），用 mlx-vlm 自带的 `mlx_vlm.speculative.drafters.qwen3_5_mtp.split` 拆成独立目录
  `~/.cache/lmk-research/qwen3.8-27b-mtp-draft`（810 MB bf16，`block_size` 3）。
- 主模型：lmk pull 的 `lmstudio-community/Qwen3.8-27B-MLX-4bit`，用引擎同样的 `mlx_vlm.utils.load_model` 加载，
  直接调 `mlx_vlm.generate.ar.generate_step`（B=1，引擎批处理路径的底层），`draft_kind="mtp"`。
- 常驻 lmk 先 `lmk down`，跑完 `lmk up`。
- 贪心、思考关、400 token、exp01 的三题各 2 遍。臂：`none` · `mtp-b2` · `mtp-b3`（默认） · `mtp-b4` · `mtp-b6`。
- 记 decode tok/s（同 bench 口径）、每轮接受数（`speculative_stats_since`）、输出是否与 `none` 逐字节相同（贪心应相同）。

## 预期（跑之前写）
- **none ≈ 39.5 tok/s**（exp01 顺序路径与批处理路径都是这个数）。把握：高。
- **成本模型**：草稿一步要读一层 bf16（0.8 GB）+ 主模型的 lm_head（4bit 约 0.6 GB）≈ 1.4 GB，主模型一步约 16 GB ⇒ 草稿一步 ≈ 主模型的 1/11。
  b=3 一轮 = 3 步草稿（≈0.27 步）+ 1 步验证 ≈ 1.3 步。若每轮平均接受 2.2 个（含验证送的 1 个）⇒ 2.2/1.3 ≈ **1.7×**。
- **story 1.5–1.8×，code 1.8–2.2×，copyedit ≥2.5×**。把握：低到中——接受率是要量的。
- **block size**：story 上 b=2/3 最好，b=6 反而略慢；copyedit 上 b=4/6 最好。把握：中。
- **输出逐字节相同**：greedy 下应相同（mlx-vlm 的 Gemma MTP 文档称 byte-identical）。若不同，是 bug 或数值差，要记。把握：中高。
- **落到判据**：预计跨过 1.8×（至少在 code/copyedit 上），结论偏向"fork 先接投机解码"；且因为 mlx-vlm 已经实现了，
  fork 的工作量是"接线"而不是"实现"。

## 怎么跑
`run.sh`（`lmk down` → `run_mtp.py` → `lmk up`）。原始 JSON 在 `raw/`，日志 `run.log`，汇总 `results.md`。
