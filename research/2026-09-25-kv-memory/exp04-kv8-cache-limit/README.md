# exp04：exp02 换成 kv8——1 GiB 缓冲池上限在 kv8 上是不是也不拖慢 decode

日期 2026-09-25（23:56 起跑）。m3u（M3 Ultra，96 GB），lmk worktree `kv-memory`，引擎 42a248c，mlx 0.32.0。常驻服务停着（控制者 2026-09-25 22:49:03 `lmk down`）。

## 为什么加这一组
exp02 只在 kv16 上量了上限；exp03 要给 kv8 也算"设上限后"的一列。kv8 的注意力路径拆成几段算子（SLC-007：反量化、量化 matmul），
每步的临时 buffer 比 kv16 多——在 kv16 上无害的 1 GiB，在 kv8 上可能让每步都重新向 Metal 要 buffer。把 exp03 的 kv8 设上限一列从"推"变成"量"。

## 条件与方法
与 exp02 相同（`run.sh` 由 exp02 的拷来，只改 `kv_cache_bits: 8` 与条件表）：code 任务，32k / 128k，每档 warm + 3 次 256 token 解码；
三个条件依次 `unset1`（不设限）→ `limit1g`（`mx.set_cache_limit(1 GiB)`）→ `unset2`；共用一个临时 cache 目录（只有 `unset1` 冷算前缀）。

## 对比基准
- exp02（kv16，同 harness）：1 GiB 上限在 128k 把解码中的 footprint 55.0 → 39.7 GB，请求后 36.5 → 27.8 GB，decode 32.8 → 32.5 tok/s（−0.8%，文本逐字相同）；
  32k 51.2 → 47.2 / 50.6 / 51.9（第一次偏低）。上限 0：128k −14%，32k −7%。
- exp01（kv8，开草稿，code + prose 中位）：128k 解码中 footprint 38.9 GB、MLX cache 最高 14.9 GB、peak 31.2 GB，请求后 28.5 GB；code 20.7 tok/s；32k 41.2 tok/s。

## 预期（跑之前写）
- **K1 内存**：1 GiB 上限让 kv8 128k 解码中的 footprint 从约 39 降到约 28–33 GB（≈ active 最大 + 1 + 1.1），请求后从约 28.5 降到约 23–24 GB。把握：中高（kv16 上同形）。
- **K2 速度**：kv8 128k decode 慢 ≤ 2%，32k ≤ 3%。把握：中（kv8 每步临时 buffer 更多，是这组存在的理由；如果 1 GiB 在 kv8 上慢了，说明量化路径每步的临时量
  超过 1 GiB 或分配模式不同，要改用 4 GiB）。
- 文本三个条件逐字相同（上限不改数值）。把握：高。
