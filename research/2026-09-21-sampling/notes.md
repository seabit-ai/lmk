# 采样参数：temperature / top_p / top_k / min_p / repetition_penalty / stop / seed

日期 2026-09-21。起因：公开前，OpenAI 兼容 server 最基本的预期；README "What does not work yet" 第一条。
裁决（owner，2026-09-21，四点）：① 名字映射 + 越界 400 + 未知参数忽略；② 不传就用模型自带的 generation_config；
③ seed 照引擎的全局语义实现、README 写明；④ stop 只对回答段生效——引擎行为实测后再定。

- **SMP-001 引擎的缺省是贪心，且从不读 generation_config.json**（`mlx_engine/utils/sampling.py:93`，THK-001 已记）。
  裁决 ② 的"模型自带"要由 lmk 读 `generation_config.json` 兑现：`temperature`→`temp`、`top_p`、`top_k`；`do_sample: false` ⇒ temp 0。
  Qwen3.8-27B 发布的值：temp 1.0 / top_p 0.95 / top_k 20。**这是 lmk 行为的改变**：之前一直贪心，现在缺省按模型作者的设置采样。
- **SMP-002 stop 交给引擎会在思考段里命中**（exp01）：回答为空、finish=stop。⇒ lmk 不把 `stop_strings` 传引擎，
  在自己的输出切分器之后、只对回答段匹配；命中即关闭引擎的生成器（取消路径现成的 `generation.pieces.close()`）。
  代价：匹配在 piece 粒度，命中前引擎可能多算几个 token，用户看不到。
- **SMP-003 seed 在 lmk 走的批处理路径上被引擎忽略**（`generate.py:688`："Seed arg is ignored for batched gen"；
  lmk 的模型走 `BatchedVisionModelKit`）。裁决 ③ 无法兑现：seed 收下、不传、记 `LmkParamIgnored` 日志，README 写明不支持。
  可复现要靠 `temperature: 0`（exp01 验证逐字相同）。
- **SMP-004 引擎参数范围**：`temp` 0 = argmax；`top_p` 只在 (0,1) 生效；`top_k` 0 = 关；`min_p` [0,1]。lmk 的校验按 OpenAI 的
  范围（temperature 0–2、top_p (0,1]、stop ≤ 4 条）。
