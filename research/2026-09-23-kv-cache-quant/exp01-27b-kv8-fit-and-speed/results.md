# exp01 结果（2026-09-23）

`raw/serve-kv{16,8}.log`（引擎 auto-fit 行）、`raw/bench-kv{16,8}.txt`、`raw/cache-size-kv{16,8}.txt`。同一 seed 7，各自空 cache。

| | kv16（模型自己的精度） | kv8 | 预期 kv8 |
|---|---|---|---|
| 探针 full_kv | 65,536 B/token | **34,816 B/token**（53.1%） | 34,816 ✔ |
| fitted ctx（96 GB） | 262,144 | 262,144 | 262,144 ✔ |
| 冷 prefill（4,074 token） | 324 tok/s | 323 tok/s | 不变 ✔ |
| cache 命中还原 | 58k tok/s，首 token 1.01 s | **47k tok/s**，首 token 1.03 s | ≥ 53k ✘ |
| decode（400 token，短 prompt） | 39.5 tok/s | 38.9 tok/s | 39.5 ±1 ✔ |
| 磁盘 cache（bench 三题） | 880 MB | 909 MB | 约 53% ✘ |

## 两个没命中的
- **命中还原慢了 19%**（58k → 47k）。还原读的字节少了，但每层要拼三个数组（packed / scales / biases）再升成批 cache，小操作多了。
  用户感受得到的数是首 token：1.01 → 1.03 s，几乎没动。先记着，优化归后面。
- **磁盘 cache 没减半，反而多 3%。** 用 persist 探针（2,946 token prompt）分记录种类量（`../notes.md` KVQ-003）：
  - `kv_delta`（16 层 full attention 的 KV）：每 256-token 块 16.00 → **8.51 MiB**，53.2%，和预期一致；
  - `state_checkpoint`（48 层 linear attention 的状态）：每个 **146.8 MiB**，两臂一样——量化碰不到它，而它是 Qwen3.8 磁盘 cache 的大头
    （这个 prompt：KV 176 MiB 对检查点 294 MiB）。
  所以 Qwen3.8 家族的磁盘 cache 只省 KV 那部分（persist 探针 470 → 387 MiB，−18%）；bench 那 3% 的增长来自另一次检查点落盘，
  不是量化记录变大。"体积减半"这条预期错在把 Qwen3.8 当成纯 KV 模型；Gemma 31B（全 KV）才会接近减半。
- 附带一条：8 位 KV 下贪心输出**在 3k prompt 上就分叉**（persist 探针第 20 个 token 起：`Looking` 对 `</think>`）。不是错，是数值；
  验收第 1 条（长上下文评测臂）才回答"有没有变笨"。

## 与验收对照
- 第 2 条（表里 ctx 由探针实测）：探针会量量化 cache，数是 34,816——**通过**。96 GB 上 27B 两臂都是上限，8 位的收益要在
  32/48 GB 档体现：按引擎公式，32 GB 上 27B 的 ctx 从约 85k 到约 150k（外推，未实测）。
- 第 3 条（cache 身份带位宽）：两臂各自的 cache 目录不同（单测锁定），bf16 记录不会被量化会话读到——**通过**。
- 第 4 条（bench 记录）：两行进 `docs/benchmarks.md`——**完成**。
- 集成测试：27B-4bit 8 位，思考开 4/4、思考关 4/4、进程重启后前缀 cache 还原 1/1——**通过**。
- 第 1 条（长上下文评测臂）：**未做**。这是写进模型页推荐之前的最后一关。
