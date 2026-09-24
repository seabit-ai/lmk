# exp02：长上下文下 8 位 / 4 位 KV cache 会不会变笨（验收第 1 条）

日期 2026-09-23。m3u。引擎 fork `lmk` d3650db，lmk main（含 `kv_cache_bits`）。27B-4bit 的推荐配置（`reasoning_effort: low`）。

## 怎么量
- 题：智能评测的 harness（`research/2026-09-23-intelligence-27b-vs-122b/harness/`）原题，取 code（HumanEval 前 20 题）、
  instruct（10 题格式/指令题）、tools（5 题 agent 任务），每题 2 遍（`EVAL_RUNS=2`，`EVAL_LIMIT=20`）。
- 每个对话前面垫 **6 万 token 的真实内容**（`prefix-60k.txt`：lmk 与引擎的源码和文档，用 27B 的 tokenizer 截到 60,000 token），
  作为 system 消息里的"参考材料"；题本身不引用它。量的是长上下文累积的量化误差会不会拖垮后面的任务。
- 三臂各起一个临时 lmk（端口 1236，空 cache）：`kv16`、`kv8`、`kv4`。先发一条带前缀的热身请求，后面的请求前缀走 cache 命中——
  这正好让量化记录的磁盘往返也在测试路径上。
- **k8v4 臂没跑**：mlx-vlm 的 `quantized_scaled_dot_product_attention` 只收一个 `bits`，K/V 分宽要改 kernel 调用层不止十行；
  先用 kv4 探底——4 位站得住，k8v4 就没必要；4 位塌了，k8v4 才值得做。
- 机器判分（原 harness），`run.py grade` 出表。常驻先 `lmk down`，跑完 `lmk up`。

## 预期（跑之前写）
- **kv16 ≈ 无前缀时的 27B-low**：code 前 20 题 ≈ 95%+，format 100%，tools 5/5。前缀不引用，模型应能忽略。把握：中——
  6 万 token 的无关材料本身可能让小模型分心，这一臂是基线，不是量化的错。
- **kv8 与 kv16 差在噪声内**：每类相差 ≤1 题。把握：中高（Ollama/llama.cpp 社区经验：q8_0 "very small loss"）。
- **kv4 明显掉**：code 掉 10% 以上，tools 可能出错（参数抄错、路径错）。把握：中。
- **耗时**：8 位不比 16 位慢多少（exp01：decode −1.5%）；4 位同理。把握：中。
- **判据**：kv8 每类不比 kv16 少 2 题及以上 ⇒ 验收第 1 条通过，27B 页写推荐 `kv_cache_bits: 8`。
