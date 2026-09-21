# 内存护栏：取证

日期 2026-09-20。为 `docs/design/2026-09-20-memory-guard.md` 供事实。编号前缀 `MG-`。
机器：M3 Ultra 96GB；模型 Qwen3.8-27B-MLX-4bit；引擎 mlx-engine `08f0c07`。

## 发现

### MG-001 引擎已经有一套细致的内存模型——但只管"这个模型 + 一条序列 + 整台机器归我"
`mlx_engine/model_kit/batched_vision/context_fit.py`：加载时用一个 token 的探针实测 cache 形状，解一个峰值公式
（baseline + 每 token 的 KV + prefill 步长相关的注意力中间量 + 固定 3GiB 余量），把窗口拟合进
`mx.device_info()["max_recommended_working_set_size"]`；请求准入时再按同一组系数排 prefill 步长（2048→1024→512）。
m3u 上对本模型的日志：`working_set=77.76GiB reserve=3.00GiB baseline=14.95GiB full_kv=65536B/token fitted=262,144`（= 模型上限，未下调）。
核算：262,144 × 64KB = 16GiB；14.95 + 16 + 3 ≈ 34GiB，远小于 77.76。

### MG-002 拟合不知道并发：lmk 以 `max_seq_nums=4` 加载，四条满长序列会超
`context_fit.py` 里没有任何 sequence / batch 维度。4 × 16GiB + 14.95 + 3 ≈ 82GiB > 77.76GiB。
只在"四个 26 万 token 的请求同时在跑"时才触发；kitten 的实际形态是一个 turn + 偶尔一个 groom（LMS-013 见过两者并发）。
两条 10 万 token 的序列 ≈ 12.5GiB，无虞。**并发本身在 lmk 上从未测过**（README 已写明）。

### MG-006 并行生成给吞吐（2 个 1.7×、4 个 2.2×）；读 prompt 严格串行，且会把正在生成的请求拖到近乎停顿（exp01）
机制（`batch_generator.py::_next`）：每圈先让所有在生成的序列一起走一步（batch），再给**一条**待处理 prompt 读**一块**（≤2048）。
实测：1 个 39.7 tok/s；2 个各 34；4 个各 22。一个 13k 的全新 prompt 进来读了 43 秒，期间另一个正在生成的请求只出了 11 个 chunk。
并发 4 个无错无卡。**MG-002 的修正**：四条序列同时*生成*是现实可达的，但"四条满长"仍只是理论上限；内存风险来自 KV 总量，
与这里量到的速度问题是两件事。

### MG-003 加载前没有任何检查
引擎里搜不到"权重体积 vs 可用内存"的判断。权重超过工作集的模型会被照常加载——LM Studio 的 guardrail 防的就是这个，
而 lmk 没有对应物。缺省模型 15GB 无此问题；`model.repo` / `model.path` 指向任意模型时才有。

### MG-004 拟合看不见别的进程
baseline = `mx.get_active_memory() + mx.get_cache_memory()`，只是**本进程**的 GPU 内存；拟合在加载时算一次。
m3u 上的邻居：oMLX 0.3.2（常驻 8000 端口，自报可用上限 79.2GB，现未加载模型）、LM Studio（guardrail=high）、STT。
它们任何一个加载大模型时，lmk 手里已经占着 19GB（`footprint`：phys 19GB，峰值 22GB），并且还会按"整机归我"的假设继续长。

### MG-005 要防的那次事故，原因其实没查明
09-19 08:13 的整机无响应（kitten repo research/2026-09-19-bruce-harness-case-study BHC-006）：关机瞬间的 stackshot 显示全机
footprint 17.5GB / 96GB，证据**不支持**"内存被吃光"。也就是说这道护栏防的是一类合理的风险，不是一个已确诊的病因——
它应当便宜。

## 未测
- 权重超过工作集的模型在 lmk 里加载会发生什么（报错？卡死？）。没敢在常用机上试。
- 并发请求下的真实峰值。
- macOS 内存压力等级（`kern.memorystatus_vm_pressure_level`）在邻居加载大模型时的变化曲线。
