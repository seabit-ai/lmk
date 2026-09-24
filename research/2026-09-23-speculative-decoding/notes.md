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
