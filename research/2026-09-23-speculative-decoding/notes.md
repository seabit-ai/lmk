# 投机解码（speculative decoding）能不能给 lmk 用

日期 2026-09-23。起因：owner "你听说过投机解码吗？…what are the candidate draft models for qw38-27B?"

- **SPD-001 当前引擎（08f0c07）在 lmk 走的路径上不支持。** `batched_vision/model_kit.py:388-397`：`BatchedVisionModelKit.load_draft_model`
  直接 raise "Speculative decoding is not currently supported for batched vision models"，`is_draft_model_compatible` 恒 False。
  实现只在顺序路径（`model_kit.py`，纯文本、无磁盘 cache）。⇒ 在这个引擎版本上，**投机解码与磁盘前缀 cache 二选一**，lmk 拿不到。
- **SPD-002 兼容判据 = 词表大小相同**（`model_kit.py:200-201`：`draft_tokenizer.vocab_size != self.tokenizer.vocab_size` ⇒ 不兼容）。
  Qwen3.8-27B 词表 248,320；同词表的小模型：Qwen3.5-0.8B / 2B / 4B / 9B（都是 qwen3_5、带 vision）、Qwen3.8-Flash-Next（qwen4_exp）。
  MLX 4bit 版在 mlx-community：Qwen3.5-2B / 4B / 9B（-MLX-4bit）。草稿模型的常规选择是主模型的 1/10 左右 ⇒ **2B 或 4B** 是候选，
  0.8B 太弱猜中率低，9B 太重。Qwen3.6-4B/9B 在 HF 上没有（404）。
- 未做：实测。要么等引擎在批处理路径补上，要么在顺序路径上量一次"没有磁盘 cache 但有投机解码"的 decode 增益作参考。
- **SPD-003 mlx-lm 的投机解码对 Qwen3.5/3.8 全家族不可用**（exp01）：它靠 `trim` 回滚 KV cache，Qwen3.8-27B 48/64 层是 linear
  attention，mlx-lm 用不可裁剪的 `ArraysCache` ⇒ 第一条请求就抛 "requires a trimmable prompt cache"。SPD-002 的候选（Qwen3.5-2B/4B）
  在 mlx-lm 路径上等于没有。顺序路径无草稿 decode 39.2–39.5 tok/s，与批处理路径相同。
- **SPD-004 Qwen3.8 自带草稿头（MTP）**：owner 提醒。原版 `Qwen/Qwen3.8-27B` config `mtp_num_hidden_layers: 1`，第 18 个分片里 15 个
  `mtp.*` 张量（一层 full attention + fc + 三个 norm，共享主模型的 embedding/lm_head），bf16 810 MB。**lmstudio-community 和
  mlx-community 的 MLX 4bit 转换都把它剥掉了**（2180 个张量里 0 个 mtp）。只下那一个分片 + 拆分即可得到草稿器。
- **SPD-005 mlx-vlm（引擎钉住的 321514d = 0.6.12）已经有投机解码和 KV 量化，只是 mlx-engine 没接**：`mlx_vlm/speculative/`
  三种草稿家族（dflash / eagle3 / mtp）、`drafters/qwen3_5_mtp/`（含 `split.py` 拆分工具）、批处理版轮回 `_mtp_rounds_batch`、
  混合注意力回滚 `rollback_speculative_cache`；`kv_quant.py` + `BatchQuantizedKVCache` / turboquant。设计文档 §2 对 B 方案的
  工作量估计（"搬实现"）作废，改为"接线"。
- **SPD-006 实测（exp02，B=1，贪心，mlx-vlm 原样）**：b3 时 story 1.22× / code 1.59× / copyedit 1.70×；接受率高（每轮 2.1–3.0 个）
  但每个草稿 token 折合 0.4 个主模型步（预期 1/11），开销是实现层面的；b2 是散文最优，b≥4 散文倒退。
- **SPD-007 贪心输出不逐字节相同**：code/copyedit 相同，story 第 189 字符起分叉（近平手词被 block 验证的数值差翻转）。进产品要定义
  可接受差异 + 测试（设计文档 §4 验收项要改：不是"逐 token 一致"，而是"代码类逐 token 一致、散文允许分叉且质量不降"）。
- **SPD-008 其他草稿器**：`z-lab/Qwen3.8-27B-DFlash2`（3.6 GB，DFlash v2）存在，这版 mlx-vlm 只写了 v1，未试。z-lab 也有
  `Qwen3.5-122B-A10B-DFlash`（对应 lmk 的 122B 两个版本），未试。
