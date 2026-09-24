# KV cache 量化（fork 上的第一块接线）

日期 2026-09-23。SAD 裁定见 `docs/design/2026-09-23-own-engine.md` §4；命名依据 `research/2026-09-23-kv-cache-naming/`。
引擎补丁：fork `seabit-ai/mlx-engine` 分支 `lmk` 的 d3650db（上游 08f0c07 + 一个 commit）；lmk 侧分支 `kv-cache-bits`。

- **KVQ-001 接线范围（实际）**：引擎 6 个文件 +111 行：`generate.py` 放行；新 `kv_quant.py`（造量化 cache、单条升批、探针辅助）；
  `batch_generator.py` prefill 用量化 scalar cache、入批时升成 `BatchQuantizedKVCache`；`records.py` / `disk_budget.py` 量化 KV 记录按 token 轴切拼；
  `context_fit.py` 探针直接量量化 cache。mlx-vlm 一行没改。引擎自己的单测靠 monkeypatch 模块级 `make_prompt_cache`，
  接线时要保留那个钩子（第一版删了 import，挂 47 个测试）。
- **KVQ-002 探针实测每 token 字节**（exp01）：27B-4bit 8 位 = 34,816 B/token = 16 位的 53.1%，与算术一致（1 B 打包 + 每 64 元素一对 bf16 scale/bias）。
- **KVQ-003 Qwen3.8 的磁盘 cache 大头不是 KV**：2,946 token prompt，KV 记录 11 × 16 MiB = 176 MiB，linear attention 状态检查点 2 × 147 MiB = 294 MiB。
  量化只动前者（→ 8.5 MiB/块）。所以这个家族磁盘 cache 只省约 18%；纯 KV 家族（Gemma 31B）才接近减半。
- **KVQ-004 速度**：冷 prefill 不变（323/324），decode −1.5%（38.9/39.5，短 prompt），cache 命中还原 −19%（47k/58k；首 token 1.03/1.01 s）。
  还原慢在每层三个数组的拼接与升批；用户看到的是首 token，几乎没动。长上下文下 decode 的差别未量。
- **KVQ-005 输出会变**：8 位下贪心输出在 3k prompt 第 20 个 token 就分叉。验收第 1 条（长上下文评测臂）才回答变不变笨。
- **KVQ-006 集成测试 8 位**：27B-4bit 思考开 4/4、关 4/4、重启还原 1/1（`LMK_ITEST_KV_BITS=8`）。
- 未做：验收第 1 条（长上下文评测臂，含 k8v4 变体）；Gemma 31B / 122B 的探针数；32/48 GB 档的实测（没有机器，公式外推）；
  README 表"推荐配置下的数"（等第一个模型过门槛）。
