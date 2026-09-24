# exp10：mlx-dspark（独立的 MLX 实现）在这台机器上跑 DFlash 2 能到多少——校验器是不是瓶颈的对照

日期 2026-09-24。m3u。起因：exp07/09 只有 1.18–1.27×，owner 问"这正常吗"。一轮账目（scratch `round_bench.py`）：普通单步 29.1 ms，
草稿一步 17.5 ms，**exact verifier 校验 T=8 要 109 ms**（T=4 64.5 ms，几乎按 token 线性），一轮 127 ms，每轮接受 4.4 ⇒ 上限 1.0×。
外面：ivanfioravanti 在 M3 Ultra 上用 mlx-dspark 报 4-bit 27B + DFlash2 约 97 tok/s（≈2.5×）。mlx-dspark 用一次普通批量前向校验。

## 怎么量
`mlx-dspark generate`（PyPI，scratch venv），`--mode baseline` 和 `--mode dflash --drafter incoai/Qwen3.8-27B-DFlash2`，同一个 4-bit 主模型，
贪心 400 token，代码题与散文题各一次；读它打印的 tok/s 与接受统计。跑时 `lmk down`。

## 预期（跑之前写）
- E1 baseline ≈ 38 tok/s（同 exp07 的普通解码）。把握：高。
- E2 dflash 代码题 ≥ 80 tok/s（≥ 2.0×），散文 ≥ 50。把握：中——97 是别人一条推文的数，条件不明。
- E3 若 E2 成立：mlx-vlm 0.6.16 的 exact verifier 就是我们 1.18× 的原因，B 的接线要绕开它（普通批量前向校验 + 自己收 GDN 状态做回滚）。

## 过程与结果（`raw/run.log`）
mlx-dspark 0.x（PyPI）自己带了几处按机器校准的优化（sdpa-split、小 M 的 qmm 内核、prefill 的 CPU 分担）；它用的是**思考开着**的模板（输出以 "We need answer user…" 开头）。

| | decode tok/s | 每轮接受 | 目标前向次数 / 400 token | 一轮 |
|---|---|---|---|---|
| baseline（普通贪心） | 39.6 / 39.7 | — | — | 25 ms |
| DFlash2，代码题 | **50.0**（1.26×） | 3.13 | 130 | 65 ms |
| DFlash2，散文题 | 49.8（1.25×） | 3.17 | 127 | 66 ms |

- **E2 落空**：这台机器上 mlx-dspark 也只有 1.26×，不是推文里的 97 tok/s（那条推文条件不明：8-bit 模型 3.63× 的宣称对应 84 tok/s；数学题每轮 5.5 也能算出 85）。
- 但它的**一轮只要 65 ms**（草稿约 18 + 普通批量前向校验约 47），我们 exact verifier 的一轮 127 ms。它每轮只接受 3.1（思考文本难猜），所以倍数和我们一样低；
  换成我们 exp09 的接受率（代码 4.4、固定块 8 时 4.9–7.4），同样的 65 ms 一轮就是 68–115 tok/s（1.75–2.9×）；散文（每轮 2.0）会**慢于**普通解码（约 31 tok/s）。
- **E3 成立**：校验器是我们的瓶颈。DFlash2 在 M3 Ultra 上的合理天花板：代码约 1.8–2×、复述约 3×、散文不赚（要按接受率自动退回普通解码）。
