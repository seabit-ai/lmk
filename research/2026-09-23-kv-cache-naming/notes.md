# 别家怎么给"KV cache 量化"起名（给 lmk 配置项定名用）

日期 2026-09-23。起因：引擎路线 SAD 第 3 点，owner "can you do some research how other framework name those?"
来源：各家官方文档/README/CLI help 原文（网页抓取），本机装的 mlx-lm 0.31.3、mlx-vlm 0.6.12、LM Studio MLX 后端 @34 的源码。

| 框架 | 在哪设 | 名字 | 取值 | 默认 | 作用域 | K/V 分开 |
|---|---|---|---|---|---|---|
| llama.cpp | CLI | `--cache-type-k` / `--cache-type-v`（`-ctk` / `-ctv`） | f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1 | f16 | 进程 | 是；草稿模型另有 `--cache-type-k-draft` |
| Ollama | 环境变量 | `OLLAMA_KV_CACHE_TYPE` | f16 q8_0 q4_0 | f16 | 全局，"所有模型都用这个" | 否 |
| LM Studio（llama.cpp 后端） | GUI / SDK | GUI "K Cache Quantization Type" / "V Cache Quantization Type"；SDK `llamaKCacheQuantizationType` / `llamaVCacheQuantizationType` | q8_0 等，或 `false` | 关 | 每次加载模型 | 是 |
| LM Studio（MLX 后端 = mlx-engine） | 引擎参数 | `kv_bits` `kv_group_size` `quantized_kv_start` | bits 2/3/4/6/8；group 32/64/128 | 关 | 每次加载 | 否 |
| mlx-lm | CLI | `--kv-bits` `--kv-group-size` `--quantized-kv-start` | 同上 | 关 | 进程 | 否 |
| mlx-vlm server | CLI | `--kv-bits`（float，3.5 = TurboQuant）`--kv-key-bits` `--kv-value-bits` `--kv-key-scheme` `--kv-value-scheme` | uniform / turboquant | 关 | 进程 | 是（turboquant 缺省 K=floor、V=ceil） |
| vLLM | CLI | `--kv-cache-dtype`（另有 `--kv-cache-dtype-skip-layers`） | auto fp8 fp8_e4m3 fp8_e5m2 | auto | 进程 | 否（按层跳过） |
| SGLang | CLI | `--kv-cache-dtype` | fp8_e4m3 fp8_e5m2 | 关 | 进程 | 否 |
| HF transformers | generate() 参数 | `cache_implementation="quantized"` + `cache_config={"backend","nbits","q_group_size","residual_length"}` | nbits 2/4/8 | 关 | 每次调用 | 否（`axis-key`/`axis-value` 是量化轴） |
| koboldcpp | CLI | `--quantkv` | f16 bf16 q8_0 q5_1 q4_0（旧版 0/1/2） | f16 | 进程 | 否（无 flash attention 时只量 K） |
| exllamav2 | 类 / CLI | `ExLlamaV2Cache_Q4/Q6/Q8`；`-cq4/-cq6/-cq8` | 4/6/8 | 关 | 进程 | 否 |

## 发现
- **KVN-001 两个命名家族**：llama.cpp 血统和 vLLM 叫 "type / dtype"，值是 `q8_0`、`fp8` 这种格式名；MLX 血统（mlx-lm、mlx-vlm、
  LM Studio MLX 后端）和 HF 叫 "bits"，值是整数。lmk 在 MLX 上，底下的参数就叫 `kv_bits`；对 MLX 用户 `bits` 是对得上号的词。
- **KVN-002 没有一家发明"用户友好"的新词**，全都直呼 "KV cache"。LM Studio 的 GUI 标签是 "K Cache Quantization Type"，
  也没翻译成人话。客户搜这个功能用的关键词就是 "KV cache quantization"。
- **KVN-003 默认全部是关**（f16 / auto / false / None）。没有一家默认开。
- **KVN-004 K/V 分开设**：llama.cpp 血统全部支持（LM Studio 把两个都暴露给用户）；MLX 血统只有 mlx-vlm 的 turboquant 支持，
  而且它缺省给 K 更少的位（floor）、V 更多（ceil）——和 KIVI 论文"K 更敏感"的经验相反，属于该实现自己的选择，没看到依据。
- **KVN-005 作用域**：Ollama、vLLM 是全局一刀切；LM Studio 是每次加载模型时设（每个模型可不同），和 lmk "推荐按模型、开关归客户"一致。
- **KVN-006 最好的一句用户文案是 Ollama 的**："q8_0: uses approximately 1/2 the memory of f16 with a very small loss in precision；
  q4_0: approximately 1/4 the memory with a small-medium loss in precision"。三个量：省多少、代价多大、对照物是什么。
- **KVN-007 mlx-lm 有个对 lmk 有用的细节**：加载已保存的 prompt cache 时校验 `--kv-bits` / `--kv-group-size` 与文件一致，不一致报错。
  磁盘 cache 身份键带位宽（验收第 3 条）有先例。

## 对 lmk 的建议（进设计文档 §4 第 3 点）
- 名字 `kv_cache_bits`，整数 16 / 8 / 4：`kv_cache` 是全行业的词，`bits` 是 MLX 家族的词。不用 `type`（我们没有 `q8_0` 那种格式名可填）。
- K/V 分开时再加 `kv_cache_value_bits`，不做 `k8v4` 字符串（没有一家这么写）。
- 注释借 Ollama 的三个量写：省多少、代价多大、对照物。
- 默认 16（关），与全行业一致；推荐按模型写在模型页。
