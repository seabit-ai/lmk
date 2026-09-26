# exp06：kv8 + 4 GiB 缓冲池上限，32k / 128k

日期 2026-09-26。m3u（M3 Ultra，96 GB），lmk worktree `kv-memory`，引擎 42a248c，mlx 0.32.0。常驻服务停着（2026-09-25 22:49:03 起）。

## 为什么加这一组
exp05：1 GiB 上限让 kv8 在 32k 上慢 7%，4 GiB 不慢——推荐值改成 4 GiB。kv16 + 4 GiB 在 128k 量过（exp02），kv8 + 4 GiB 只量过 32k（exp05）。
exp03 的每档 Mac 表要一条 kv8 设上限的实测线，这里补 128k。

## 条件与方法
exp04 的 `run.sh` 拷来，只改条件表：`unset`（不设限）→ `limit4g`；code，32k / 128k，warm + 3 次 256 token 解码；共用一个临时 cache 目录（`unset` 冷算前缀）。

## 预期（跑之前写）
- 128k 解码中 footprint：不设 39.0（exp04）→ 4 GiB 约 35–37 GB；请求后 28.5 → 约 26–27（缓冲池 5.6 GB 截到 4 GiB 以下）。decode 与不设限差 ≤ 1%。把握：中高。
- 32k 与 exp05 相同：43.5 tok/s 上下，footprint 24.2。把握：高。
