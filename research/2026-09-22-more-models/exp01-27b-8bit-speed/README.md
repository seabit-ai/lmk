# exp01：Qwen3.8-27B-MLX-8bit 的 prefill / decode 速度（对照 4bit）

日期 2026-09-22。m3u（M3 Ultra 96GB），lmk 分支 `model-27b-8bit`，另起一个 lmk（临时 LMK_HOME，端口 1236，空 cache），
常驻的 4bit 服务空闲不停。单请求。

## 预期（跑之前写）
- **decode**：受内存带宽限制，权重 29.5 GB 对 16.1 GB ⇒ 约 4bit 的 55%：**18–20 tok/s**（4bit 实测 33）。把握：高。
- **冷 prefill**：算力受限，量化位宽影响小 ⇒ **250–320 tok/s**（4bit 320）。把握：中。
- **加载时间**：约 4bit 的 1.8 倍。把握：中。

## 怎么跑
`run.sh`：起临时 lmk → 一个 2.7k token 冷 prompt（记 ttft）→ 一个短 prompt 要 300 token 输出（记 decode）→ 停。

## 结果（2026-09-22；raw/ = 8bit，raw-4bit/ = 同条件 4bit 对照，`run-4bit.sh`）
| | 加载 | 冷 prefill（4,060 token） | decode（400 token，短 prompt） |
|---|---|---|---|
| 27B-8bit | 18 s | 318 tok/s | **22.9 tok/s** |
| 27B-4bit | 16 s | 325 tok/s | 39.3 tok/s |

- decode 预期 18–20，实得 22.9：**猜低了**——按权重字节比（55%）外推，实际是 58%，且 4bit 在这个条件下是 39 而不是我引用的 33
  （33 来自真实 agent 请求的日志中位数，prompt 大得多；短 prompt、空 cache 的 decode 更快——两个条件两个数，并列）。
- prefill 预期 250–320，实得 318：命中（算力受限，位宽无关）。
- 加载预期 1.8×，实得 1.1×：**猜错**——页缓存热（权重刚被 itest 读过），量的不是冷加载。
- 两次 4,060 token 的冷 prefill 都 12.5–12.8 s；启动后第一条请求。
