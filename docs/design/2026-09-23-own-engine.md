# 引擎路线：fork mlx-engine，先量后建

> 2026-09-23 SAD 开题。owner 提出"我们自己写一个 mlx-engine"；我的判断是"目标同意，路径是 fork + 补丁，不是从零写"。
> 本文记已摆出的事实、双方立场、待裁的点。**未裁**——等评测流水线跑完、投机解码收益量出来再继续。
> 取证：`research/2026-09-23-speculative-decoding/`（SPD-001/002）。

## 0. owner 要的结果
lmk 的引擎路线图归自己：**投机解码与磁盘 cache 同时有，KV 量化**——owner 的原话："those are already critical enough"。

## 1. 已摆出的事实
- 栈：lmk → mlx-engine（LM Studio，1.3 万行 Python，调度层，不含模型定义）→ mlx-vlm（模型定义，个人项目 Blaizzy/Prince Canuma）
  与 mlx-lm（苹果 ml-explore；lmk 只用它的分词器加载与工具调用解析器）→ MLX（苹果，C++ + Metal 内核，Python 只是绑定）。
- mlx-engine 两条路径：老 `ModelKit`（顺序、内存 cache、**有投机解码**）与新 `BatchedVisionModelKit`（连续批处理、**磁盘 cache**，
  2026-07 加入，5,500 行）。新路径对投机解码与 KV 量化都直接 raise "not supported"。**二选一是同一引擎两代代码各缺一块，不是原理互斥。**
- mlx-engine 单一主要作者（111/195 提交），最近提交 2026-08-21——上游节奏是真实风险。
- lmk 已经在做的事等于"轻 fork"：锁提交号 + 换掉 cache 存储类。
- 投机解码兼容判据 = 词表相同；Qwen3.8-27B（248,320）的草稿候选：Qwen3.5-2B / 4B（mlx-community 有 4bit）。

## 2. 两个方案
| | A. 从零写引擎 | B. fork mlx-engine，在批处理路径上补两项 |
|---|---|---|
| 投机解码进批处理路径 | 全部重写 | 搬老路径的实现；难点在批内序列接受数不齐、磁盘 cache 块的回滚；估一两千行含测试 |
| KV 量化 | 全部重写 | 接 mlx-lm 的量化 KV cache 进批处理注意力，磁盘存取认量化数组；估几百到一千行 |
| 拟合公式、调度、图片跨块 | 重来，且 3 GiB 预留那类校准值要重测 | 沿用 |
| 底层 | 仍是 mlx-vlm | 仍是 mlx-vlm |
| 上游死了 | 无影响 | fork 即我们的引擎 |
| 上游活着 | 白做 | 补丁可回合 / 上游追平就删补丁 |
我的立场：**B**。A 只在"架构做不到我们要的事"（按请求思考开关、多模型常驻）或"上游停更半年以上"时才值得；两者都没到。

## 3. 先量后建（下一步，已定做法）
在老路径上量投机解码的真实收益：27B-4bit + Qwen3.5-4B 草稿（另试 2B），bench 的 decode 探针，没有磁盘 cache 但 decode 数字是真的。
- 若 ≥ 1.8×：投机解码先做。
- 若 ≤ 1.3×：KV 量化先做（32 GB 档同内存下两到四倍上下文，收益确定）。
- 中间：开会再定。
条件：评测流水线跑完、GPU 空出来。结果进 `research/2026-09-23-speculative-decoding/expNN`。

## 4. 待裁的点（SAD 续场）
- fork 放哪：`seabit-ai/mlx-engine` 还是 lmk 仓库内 vendor；补丁如何跟上游（rebase 队列 vs 长期分支）。
- 先做哪个（由 §3 的数定）。
- 验收：现有 itest + cache-compat + bench 三件套之外，投机解码要加"输出与不用草稿逐 token 一致"的测试。
