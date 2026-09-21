# exp01 — 并行到底给不给吞吐？别人读 prompt 时我的生成会怎样？

日期 2026-09-20。为内存护栏 SAD 的设计点 B（并发）供事实。对象：m3u 上常驻的 lmk（build 77fcdc1，Qwen3.8-27B-4bit，`max_seq_nums=4`）。

## 读代码得到的机制（`batched_vision/batch_generator.py::_next`）
一个循环，每圈：① 所有在生成的序列一起走一步 decode（batch：权重读一遍，每条序列各出一个 token）；
② 若还有名额，给**一条**待处理 prompt 读**一块**（≤2048 token）。prefill 严格串行、先到先读；decode 与 prefill 逐圈交替。

## 方法
`run.py`：N 个线程同时发流式请求（短 prompt、`max_tokens=300`，让模型数数以保证写满），各自记 TTFT 与生成速率
（completion_tokens ÷ (总时长 − TTFT)）。
- M1：1 个请求（基线）
- M2：2 个同时
- M3：4 个同时
- M4：A 先开始生成；2 秒后 B 带一个约 8k token、从没见过的 prompt 进来。看 A 在 B 读 prompt 期间的速率。

## 预期（跑之前写的）
- M1：约 33 tok/s（此前多次测得）。
- M2：每个 28–33 tok/s，合计约 1.8×。理由：单序列 decode 受内存带宽限制（每个 token 要把 15GB 权重读一遍），batch 2 几乎白送。把握 75%。
- M3：每个 22–30 tok/s，合计 3× 左右。把握 60%（混合架构的线性注意力层 batch 后未必这么线性）。
- M4：B 读 prompt 的约 25 秒里，A 每圈只出 1 个 token、每圈夹一块约 6 秒的 prefill → **A 掉到 1 tok/s 以下**；B 读完后两者各回到约 30。把握 80%。
- 任何一项若出现报错/卡死，本身就是发现（并发在 lmk 上从未测过）。

## 结果
（跑完填）
