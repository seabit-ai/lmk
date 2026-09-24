# exp13：本机的校验代价曲线，以及一轮投机的钱花在哪

日期 2026-09-24。m3u（M3 Ultra 60 核 GPU，96 GB）。mlx 0.32.0、mlx-vlm 0.6.16，引擎 fork 3b493b5（exp12 那版）。
主模型 `lmstudio-community/Qwen3.8-27B-MLX-4bit`（4 位 group 64；64 层，16 层 full attention，24 个 q 头 / 4 个 kv 头 ⇒ GQA 6）。

## 目的
外部调研（SPD-013..017）说 Apple GPU 上 4 位模型一次校验 T 个 token 的代价按 T 近似线性涨（mlx#4265：80 核 M3 Ultra 上 S=1 29.7 / S=4 45.4 / S=8 77.1 ms），
有一个开源小批量内核能把 S=8 压到 43 ms；32k 上下文时 T ≥ 6 的注意力会掉出融合向量 SDPA（`q_len × GQA ≤ 32`）。exp12 的 DFlash2 一轮约 73 ms 是我从总数反推的。
要回答：①本机的曲线长什么样；②那个内核在本机、我们的路径上值多少、输出对不对；③32k 上有没有 T=6 的悬崖；④exp12 一轮里草稿 / 校验 / 回滚 / 其它各几 ms。
这张表决定：接不接小批量内核、后缀草稿一轮能吃多长、要不要做 SDPA 切块。

## 方法
**A（`curve.py`）纯前向曲线。** mlx-vlm 的 `load_model` 直接载模型，语言模型自己的 `make_cache()`（非批处理 cache，和引擎批处理路径的注意力掩码不同，见 B 对照），
填充文本 prefill 到 4,096 / 32,768 token。然后对 T ∈ {1..10, 12, 16} 各做 3 次预热 + 15 次计时：一次前向喂 T 个 token（含 lm_head 出全部 T 个 logits，DFlash 校验就是这样），
`mx.eval` 下一 token，**下一次的输入取上一次的 argmax（依赖链，不排队）**，每次之后把 cache 还原（KV 层 `trim`，GDN 层换回快照的状态数组）。报中位数与 p10/p90。
三臂：`stock`；`fast`（`fast_qmm.enable()`，M 6–8 走内核）；`fastwide`（再加 `MLXLM_FAST_QMM_WIDE=1`，M 9–16 走双 tile）。内核是 `fast_qmm.py`，逐字拷自
avlp12/mlx-lm d9139d3d5a（MIT）。另做一致性：同一 cache 状态下 T ∈ {6, 8, 12, 16}，stock 对 fastwide 的 logits 最大差、逐位置 argmax 是否相同。
**B（`rounds.py`）引擎里的一轮。** exp12 的路径（`load_model(max_kv_size=32768, max_seq_nums=2)` + `load_draft_model`），贪心 400 token，exp01 三题。
草稿器 × 内核臂：{MTP 头, DFlash2} × {stock, fastwide}：不插桩的 tok/s + sha（端到端）；再插桩跑一遍 code：包住草稿（`_mtp_draft_block_active` / `draft_block`）、
`_verify_block`、`rollback_speculative_cache`、整轮（`speculative_round` / `dflash_round`），每段前后 `mx.synchronize` 并 eval 该段的输出（回滚 eval GDN 状态），记每轮的校验宽度。
插桩会打断流水，插桩那遍的 tok/s 不作数，只用来分账。

## 预期（跑之前写）
- A1 T=1、4k：25–28 ms（普通解码 39.5 tok/s ⇒ 一步 25 ms，这里多一个完整 lm_head 与 python 开销）。把握：高。
- A2 stock 近似线性：4k 下 T=4 38–46 ms、T=8 52–65 ms、T=16 90–120 ms。本机 60 核，比 #4265 的 80 核算力少 25%，但 exp12 反推的约 55 ms（宽 5–9）比他的 77 ms 低，
  两个方向互相打架；我押本机 T=8 ≈ 60 ms。把握：中低。
- A3 `fast`：T=6–8 降到 T=4–5 的水平（约 42–46 ms），T=6 收益最小；T ≤ 5 不变（分发不进内核）。`fastwide`：T=9–16 贴着 T=8 的水平（±10%）。把握：中。
- A4 一致性：argmax 逐位置全同；logits 最大差 < 0.25（bf16 累加顺序不同）。把握：中。猜错的一侧：个别位置 argmax 平手翻转。
- A5 32k：T=1 比 4k 多 2–5 ms（16 层注意力读 32k 的 KV）；T=5 → T=6 有一个台阶，比 4k 上同一步多出 ≥ 10 ms（16 层 × 慢路径）。把握：低——只有 M4 Pro 的单次调用数（2.48 → 7.09 ms）。
- B1 DFlash2 code 一轮：草稿 15–20 ms、校验 45–55 ms、回滚 + 其它 ≤ 8 ms，合计约 73 ms（exp12 反推）。把握：中低。
- B2 MTP code 一轮：草稿（3 步）6–10 ms、校验（宽 4）38–44 ms、合计约 50 ms。把握：中低。
- B3 fastwide 端到端：DFlash2 code 59.4 → 64–70 tok/s，copyedit 72 → 80+；MTP 不变（宽 4 不进内核）。输出 sha 与 stock 相同。把握：中低。
- 判据：内核让 DFlash2 端到端 ≥ +10% 且贪心输出与 stock 一致 ⇒ 提议进 fork（作为一个补丁 commit，缺省开）；32k 上 T ≥ 6 的台阶 ≥ 20% ⇒ SDPA 切块值得做；
  B 里草稿段 ≥ 一轮的 25% ⇒ DFlash2 的草稿器本身也是瓶颈，后缀草稿（零成本）的相对价值更高。

## 运行
`run.sh`（`lmk down` → `curve.py` → `rounds.py` → `lmk up`）。原始输出 `raw/`。

## 过程与结果
（跑完补）
