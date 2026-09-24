# exp01：Qwen3.8-27B-4bit 开 8 位 KV cache——探针量到的每 token 字节、拟合的 ctx、bench 三个数

日期 2026-09-23。m3u（M3 Ultra 96GB）。引擎 = fork `seabit-ai/mlx-engine` 分支 `lmk` 的 d3650db（上游 08f0c07 + 批处理路径 KV 量化补丁），
lmk 分支 `kv-cache-bits` 880e469。验收第 2 条（表里的 ctx 数字由探针实测）和第 4 条（bench 三个数按条件记录）。

## 怎么量
`run.sh`：两臂 `kv16`、`kv8`，各起一个临时 lmk（临时 LMK_HOME、端口 1236、空 cache），从 serve.log 抓引擎的
"Model context auto-fit" 一行（`full_kv=…B/token`、`fitted=…`），跑 `lmk bench --url … --seed 7`（同一 prompt，各自冷 cache），停。
常驻（122B-48GB）先 `lmk down`，跑完 `lmk up`。

## 预期（跑之前写）
- **full_kv**：16 位 65,536 B/token（已知）；8 位 = 每元素 1 字节打包 + 每 64 个元素一对 bf16 scale/bias（0.0625 B/元素）= 1.0625 B/元素，
  即 16 位的 53.1% ⇒ **34,816 B/token**。把握：高（纯算术，探针量的是分配的 buffer）。
- **fitted ctx（96 GB）**：两臂都是 262,144（模型上限，16 位已经装得下）。8 位的收益在 32/48 GB 档，这台机器上量不到，用公式算：
  32 GB 上 27B 从约 85k 到约 150k。把握：中（公式那一项外推）。
- **冷 prefill**：不变 ±5%（prefill 时顺手量化 KV，成本小）：约 320 tok/s。把握：中高。
- **cache 命中**：还原读的字节减半，可能更快；但还原不只受读盘限制 ⇒ **≥ 53k tok/s**，不猜上限。把握：低。
- **decode**：bench 的 decode 题 prompt 很短，KV 读量本来就小，量化 SDPA kernel 的差别测不出来 ⇒ **39.5 ±1**。把握：中高。
  长上下文下 decode 的差别是另一题（见 backlog）。
- **磁盘 cache 体积**：同一 prompt 的记录约 16 位的 53%。把握：高。
