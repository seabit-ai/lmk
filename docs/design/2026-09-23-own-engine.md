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
- ~~投机解码兼容判据 = 词表相同；Qwen3.8-27B 的草稿候选：Qwen3.5-2B / 4B~~ **作废**（09-23 exp01，SPD-003）：mlx-lm 的投机解码
  对 Qwen3.5/3.8 的混合注意力不可用（cache 不可裁剪）。
- **mlx-vlm 在引擎钉住的版本里已经实现了两件事**（09-23，SPD-005）：投机解码（dflash / eagle3 / mtp 三种草稿器，含批处理轮回和
  混合注意力回滚）和 KV 量化（批处理 cache，uniform / turboquant）。mlx-engine 的批处理路径只是没接。
- **Qwen3.8 自带一层 MTP 草稿头**（SPD-004），MLX 转换都剥掉了；下原版一个分片、用 mlx-vlm 的拆分工具即得 810 MB 草稿器。
  实测（SPD-006）B=1 贪心：散文 1.22×、代码 1.59×、复述 1.70×；贪心输出散文不逐字节相同（SPD-007）。

## 2. 两个方案
| | A. 从零写引擎 | B. fork mlx-engine，在批处理路径上补两项 |
|---|---|---|
| 投机解码进批处理路径 | 全部重写 | ~~搬老路径的实现，估一两千行~~ **接线**：mlx-vlm 已有批处理版 MTP 轮回和混合注意力回滚；要做的是 BatchedVisionModelKit 加载草稿器、批生成循环改走 `run_speculative_rounds`、磁盘 cache 块与回滚对齐（09-23 修正） |
| KV 量化 | 全部重写 | ~~接 mlx-lm 的量化 KV cache~~ **接线**：mlx-vlm 已有 `BatchQuantizedKVCache`/turboquant；要做的是引擎批处理路径传 `kv_bits`、拟合公式认量化后的每 token 字节、磁盘存取认量化数组 |
| 拟合公式、调度、图片跨块 | 重来，且 3 GiB 预留那类校准值要重测 | 沿用 |
| 底层 | 仍是 mlx-vlm | 仍是 mlx-vlm |
| 上游死了 | 无影响 | fork 即我们的引擎 |
| 上游活着 | 白做 | 补丁可回合 / 上游追平就删补丁 |
我的立场：**B**。A 只在"架构做不到我们要的事"（按请求思考开关、多模型常驻）或"上游停更半年以上"时才值得；两者都没到。

## 3. 先量后建（已量，2026-09-23）
原计划：老路径上 27B-4bit + Qwen3.5-4B 草稿。**走不通**（exp01：mlx-lm 不支持混合注意力回滚）。改量 mlx-vlm 的原生 MTP 草稿（exp02）：
- 判据：≥ 1.8× 投机解码先做；≤ 1.3× KV 量化先做；中间开会。
- 实得（B=1、贪心、mlx-vlm 原样、block 3）：**散文 1.22×，代码 1.59×，复述 1.70×**——灰区。接受率高（每轮 2.1–3.0 个），
  拖后腿的是每个草稿 token 折合 0.4 个主模型步的实现开销（预期 1/11）。压掉这块开销是 fork 之后的优化项。
- 判据设立时的前提变了：两件事都不用"实现"，只用"接线"（§2 修正）。先做哪个不再由工作量定，由用户价值定——见 §4。
详见 `research/2026-09-23-speculative-decoding/exp02-27b-native-mtp/results.md`。

## 4. 待裁的点（SAD 续场）
- fork 放哪：`seabit-ai/mlx-engine` 还是 lmk 仓库内 vendor；补丁如何跟上游（rebase 队列 vs 长期分支）。
- 先做哪个：§3 的数落在灰区，且两件都是接线。我的倾向：**KV 量化先**——收益确定（32 GB 档同内存两到四倍上下文，直接改
  README 那张表的 ctx 列），无输出差异问题；投机解码 1.2–1.7× 且散文输出会变，还要先解决"lmk pull 怎么带草稿器"（MLX 转换里
  没有，要自己拆或自己发布一份）。待 owner 裁。
- 验收：现有 itest + cache-compat + bench 三件套之外，投机解码的一致性测试不能写成"逐 token 一致"（SPD-007：散文会分叉）；
  改为"代码/复述类逐 token 一致，散文类允许分叉但 exp 的判分不降"。KV 量化要加"长上下文下评测分不降"。
- 草稿器的发行：lmk 的模型是 `lmk pull` 的那份下载；草稿器是我们从原版权重拆出来的，要么 lmk pull 顺手多下一个分片现场拆
  （3 GB 下载换 810 MB），要么 Seabit 在 HF 发一份拆好的。
