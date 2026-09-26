# KV 精度与内存：kv8 / kv16 实测并排、MLX 缓冲池上限、按 Mac 内存自动选精度

日期 2026-09-25 / 26。编号 KVM-xxx。条件（下文都是，除非另注）：m3u（M3 Ultra 60 核 GPU，96 GB），`qwen3.8-27b-4bit` + dflash2 草稿（4 位），
思考关，贪心，**单请求**，前缀是公开源码（spec-long-context 的 `prefix.py`），decode 256 token 且前缀已在磁盘 cache；lmk main 8e0c13a、引擎 42a248c、mlx 0.32.0。
常驻服务全程停着（owner 同意，控制者 2026-09-25 22:49:03 `lmk down`）。"footprint"= 进程的 phys_footprint（活动监视器"内存"一列），GB = 10^9。

## 目的
1. 补量 kv8 的进程内存，与 kv16 同一套读数并排（spec-long-context exp01 没记，exp02 只有 kv16）。
2. MLX 的缓冲池（释放了但留着复用的 Metal buffer）在 128k 时约 10+ GB：`mx.set_cache_limit` / 请求结束 `mx.clear_cache()` 能不能还回来而不拖慢 decode。
3. 由 1、2 推出"按 Mac 内存自动选 KV 精度"的档位线。

## 发现
- **KVM-001 128k 解码时 kv8 比 kv16 少占 16 GB，KV 本身只差 4 GB**（exp01）：footprint 解码中 kv8 / kv16 = 20.7 / 22.1（8k）、24.4 / 28.4（32k）、
  29.1 / 36.2（64k）、**38.9 / 55.1（128k）**；请求结束 5 s 后 19.7 / 20.6、21.6 / 23.6、23.9 / 28.1、28.5 / 36.7；MLX 每请求 peak 18.3 / 19.1、20.9 / 23.9、24.3 / 30.4、31.2 / 43.3；
  KV 字节 0.3 / 0.6、1.2 / 2.2、2.3 / 4.3、4.6 / 8.6。引擎窗口两边都是 262,144，lmk 的 tokens-in-memory 上限 1,844,474 / 979,877。把握：高（每格 6 次，内存读数几乎不变）。
- **KVM-002 一个在内存里的 token 实际占 3–4 份 KV**（exp01 + exp02 的逐请求时间线）：请求开始时 active = 权重与草稿 17.1 GB + 上一请求留下的热 cache（约 1 份 KV）；
  从磁盘还原长前缀的 1–2 s 里 active 再冲高约 2 份 KV（MLX peak 斜率两种精度都是 3.0 × KV B/token）；这些临时 buffer 随后进缓冲池，引擎在 prefill 结束时清掉；
  解码时缓冲池 < 1 GB；请求末尾落盘快照又放进约 1.1 份 KV，一直留到下一个请求。不设上限时 footprint 斜率 4.1–4.3 × KV。3 份 KV 具体是引擎哪几处拷贝，没追到代码。
  把握：高（量）/ 中（拆解，来自 0.25 s 采样的时间线）。
- **KVM-003 lmk 现在的内存读数看不见 KV**（exp01 冒烟与全程）：`lmk/memory.py` 的 `resident_bytes`（`ps -o rss`）只看得见 mmap 的权重，
  128k kv16 解码时 RSS 27 GB、footprint 55 GB；一个 2 GB 的 mx 数组 RSS 只涨 40 MB（`footprint` 工具与 `proc_pid_rusage` 的 phys_footprint 一致）。
  它现在只用于加载进度，没有错；拿它当"进程占多少"会低估到一半。status 的 `lmk_gpu_bytes`（MLX active + cache）与 footprint 差约 1.1 GB。把握：高。
- **KVM-004 引擎的 baseline 不含草稿器**（exp01 日志）：context-fit 在草稿器加载之前跑，baseline 14.95 GiB（16.05 GB），加载完 active 17.14 GB，差 1.1 GB 不在任何公式里。把握：高。
- **KVM-005 4 GiB 缓冲池上限：128k 省 3–13 GB，decode 不变**（exp02 kv16、exp06 kv8，输出逐字相同）：
  kv16 128k 解码中 55.0 → 42.3 GB、请求后 36.5 → 31.0，decode 32.8 → 32.65 tok/s（−0.4%）；kv8 128k 39.0 → 36.2、28.4 → 26.9，−0.5%；32k 两种精度都是 0%（exp05，每格 10 次）。
  更小的上限省得更多但有代价：**1 GiB 让 kv8 在 32k 慢 7%**（40.6 对 43.5，每次在 36–41.5 间跳；kv16 −1%；128k 两种精度 −1%）；**0 让 kv16 慢 14%（128k）/ 7%（32k）**。
  请求结束 `clear_cache` 只管请求后（36.5 → 26.8），请求中照旧 53.6。TTFT（从磁盘还原）各条件无差。MLX 的缺省上限是 97.9 GB（本机内存的 95%），即不设限。
  把握：高（128k 每格 3 次，32k 10 次；1 GiB 在 kv8 32k 上为什么慢没查）。
- **KVM-006 引擎的窗口公式没算"从 cache 还原长前缀"**（exp03，由 96 GB 实测外推）：公式按每 token KV + 34.8 kB 留（kv8 69.6 kB、kv16 100.4 kB），
  实测单请求最坏占用（4 GiB 上限下）kv8 126 kB、kv16 189 kB / token。所以窗口由内存决定的 Mac 上，一个用满窗口、前缀来自 cache 的请求会超出 GPU working set：
  32 GB 超约 8 GB、48 GB 超 11–20 GB（设了上限也超；主体是瞬时 active，不是缓冲池）。96 GB 上窗口封顶 262k，所以本机碰不到。
  tokens-in-memory 上限同样按 1 × KV/token 计，在 96 GB 上从不起作用（2 × 262k < 0.98M），在小内存上会高估 3–4 倍。把握：中（外推；别的机器、两个并发请求、MLX 在内存紧时自己回收缓冲池的行为都没量）。
- **KVM-007 自动选精度的档位线（设 4 GiB 上限的前提下）：96 GB 及以上 kv16，64 GB 及以下 kv8**（exp03）：一个请求能真用到的上下文（拟合线碰到 working set 处）
  kv8 / kv16 = 32 GB 62k / 44k，48 GB 173k / 117k，64 GB 262k / 191k，96 GB 262k / 262k。不设上限时 96 GB 的 kv16 只到约 236k（本机现行配置，外推）。
  这条线也说明：在 64 GB 以下，窗口本身应该比公式给的小（kv8：32 GB 约 62k 而不是 123k，48 GB 约 173k 而不是 262k）。把握：中（算的，只有 96 GB 是量的）。

## 没量的
- 96 GB 以外的机器（exp03 全是外推：每 token 系数与机器无关、GPU 份额 81%、缓冲池只随上下文走——三条都没核）。
- 两个并发请求的内存（tokens-in-memory 上限真正起作用的场景）；262k 的真实请求；冷 prefill 在小内存上（引擎会降步长）的峰值。
- MLX 在接近 working set 时自己回收缓冲池（头文件里有 `gc_limit_`，行为未在本机验证）——若它可靠，"不设上限"在小内存上的代价是挤压别的程序而不是 OOM。
- footprint 是 0.5 s 轮询，采不到还原前缀那一瞬的最高值（用 MLX peak + 1.1 GB 补）；kv8 128k 设 1 GiB 时 footprint 比 MLX active + cache 高约 4 GB，来源没查。
- 思考开、带 tools、采样；1 GiB 上限在 kv8 32k 上慢 7% 的原因。

## 把握
KVM-001、003、004 高；002 高（量）/ 中（拆解）；005 高；006、007 中（外推）。数在 `exp01-kv8-vs-kv16-memory/`、`exp02-cache-limit/`、`exp03-per-mac-size/`（计算）、
`exp04-kv8-cache-limit/`、`exp05-cache-limit-32k-reps/`、`exp06-kv8-4g-limit/`。

## 给 owner 的待裁（不是开工令）
1. lmk 在加载引擎前 `mx.set_cache_limit(4 GiB)`（KVM-005）：本机 kv16 128k 时少占约 13 GB，decode 不变。
2. 按内存选 KV 精度：≥ 96 GB kv16、≤ 64 GB kv8（KVM-007）；模型页分档。
3. 小内存上的窗口与 tokens-in-memory 上限要把"还原前缀约 3 × KV"与草稿器算进去（KVM-004、006）——在能量到一台 32/48/64 GB 机器之前，是先按外推收紧还是先保持引擎公式。

## 裁决（owner，2026-09-25，"y"）
- 待裁 2 通过：`model.kv_cache_bits` 缺省 = 按内存自动（≥ 96 GB 16 位，以下 8 位），config 里写的值永远优先。落地：`models.auto_kv_cache_bits`
  （常量 `KV16_FROM_MAC_GB`），种子 config 写 `kv_cache_bits: auto`，`lmk status` / bench / LmkReady 标出"自动还是配置"。
- 落地时的一处收窄（待 owner 确认）：96 GB 以下只对**以 8 位过了进表门槛**的模型选 8（`TestedModel.kv8_tested`，今天只有 qwen3.8-27b-4bit），
  其余模型与 repo/path 来源仍是 16——kv8 在它们身上没跑过验收。放开 = 删掉那个条件。
