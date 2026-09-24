# exp13：DFlash 2 草稿器 4 位版在引擎路径上——和 bf16 一样吗

日期 2026-09-24。m3u。fork `lmk` 3b493b5。草稿器 `~/.cache/lmk-research/qwen3.8-27b-dflash2-4bit`（`mlx_vlm.convert -q --q-bits 4`，4.5 bit，1.08 GB），
要发布成 `seabit-ai/Qwen3.8-27B-DFlash2-4bit`。对照 exp12（bf16，同引擎、同脚本）。

## 预期（跑之前写）
- E1 与 bf16 版速度持平（±5%），code 约 1.5×、copyedit 约 1.8×（exp07：4 位不改速度，开销在计算）。把握：中高。
- E2 每轮接受与 bf16 持平（exp07：4.30 对 4.26）；code / copyedit 与普通解码一致（kv16）。把握：中。

## 过程与结果（`raw/smoke-kv16.txt`、`raw/smoke-kv8.txt`）

| 贪心 400 token | bf16（exp12） | **4 位** | MTP 头（exp11） |
|---|---|---|---|
| kv16 story / code / copyedit | 1.01× / 1.51× / 1.83× | **1.18× / 1.71× / 2.05×** | 1.20× / 1.53× / 1.51× |
| kv16 code tok/s | 59.4 | **67.6** | 60.6 |
| kv8 story / code / copyedit | 0.98× / 1.54× / 1.84× | **1.16× / 1.73× / 2.08×** | 1.15× / 1.53× / 1.51× |
| 轮数 · 接受/草稿（kv16） | 486 · 845/1212 | 492 · 839/1219 | 566 · 764/1129 |
| 采样 temp 1.0 code 接受率 | 81% / 78% | 73% / 78% | 89% / 82% |

一致性与 bf16 完全相同（kv16 code/copyedit 一致、story 分叉；kv8 code 分叉，sha 也相同）。`lmk bench`（owner 机器，kv8，`draft: dflash2`）：code 68.9、prose 45.1，金丝雀过。

- **E1 猜错，好的一侧**：4 位快了 13%（code）到 17%（story）。exp07 说"4 位不改速度"是在 mlx-vlm 0.6.16 自己的循环里量的——那里 exact verifier 占一轮的大头，
  读草稿器权重的时间被淹没；引擎现在用普通前向校验（exp11），一轮便宜了，读 4 GB 对 1.1 GB 就显出来了。**exp07 的那条结论只在它的条件下成立**。
- E2 命中：每轮接受、一致性都与 bf16 相同。
- 结论：在 M3 Ultra 上，4 位 DFlash2 在 code（1.71 对 1.53）和 copyedit（2.05 对 1.51）上明显赢 MTP 头，story 打平（1.18 对 1.20），草稿器 1.1 GB 对 0.8 GB。
