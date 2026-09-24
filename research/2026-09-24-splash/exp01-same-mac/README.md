# exp01：Splash 和 lmk 同一台机器、同样的探针

日期 2026-09-24。m3u（M3 Ultra 96 GB，macOS 26.6.2）。Splash 装法 `brew install incoai/tap/splash`，模型 `incoai/Qwen3.8-27B-Splash`（4-bit + DFlash 2 草稿，17.4 GB）。
lmk：main 84bea38 部署版，`qwen3.8-27b-4bit` + `kv_cache_bits: 8` + `speculative_decoding: true` + `thinking: false`（今天 `lmk bench` 的行在 `docs/benchmarks.md`）。
两边都是 Qwen3.8-27B 4-bit；量化配方不同（他们自己打包的），不是同一份权重。

## 怎么量（`probe.py`，两边同一个脚本，只换 URL 和 model id）
一次只跑一个服务（量 Splash 时 `lmk down`，量 lmk 时 `splash` 停掉），temperature 0，HTTP 客户端计时：
1. `lmk bench` 同款四探针：暖身；冷 prefill（~4k token 随机 nonce）；同 prompt 再来一遍（命中，首 token 时间）；decode 散文 / 代码各 400 token。
2. 长命中：~32k token 的 prompt 冷一遍、再命中一遍，记两次首 token 时间（对 SPL-005 的 "282 ms"）。
3. 并发 4 条短 prompt 各 400 token，合计 tok/s（对 SPL-005 的 "170"）。
4. 重启服务后再发第 2 条的 prompt：命中还在不在（对 SPL-004"没说能跨重启"）。
Splash 的流式 usage 不一定带 `cached_tokens`，命中判定用首 token 时间。

## 预期（跑之前写）
- E1 decode：Splash 代码 > lmk 代码（58.6）。DFlash 2 专训草稿 + 专用内核，他们在 M5 Pro 上 74 tok/s 短 prompt；M3 Ultra 带宽更高。猜 80–100。把握：中。
  散文也应高于我们的 44.5（我们散文接受率只有 53%）。
- E2 冷 prefill 4k：接近，Splash 略高（他们 32K prompt 360 tok/s 对 oMLX 110；我们 321）。猜 350–450。把握：中低。
- E3 4k 命中首 token：Splash 明显快（GPU 内存里的 cache 对我们的磁盘还原 + 256 块尾巴）：猜 < 0.3 s 对我们 0.9 s。把握：中高。
- E4 32k 命中：Splash 亚秒；lmk 1–2 s（27k 会话实测每步 0.8–1.5 s）。把握：中高。
- E5 4 并发合计：Splash 高于 lmk（我们投机只在单行、2 并发短 prompt 1.7×，4 并发 2.2×≈ 87 tok/s 普通解码）。猜 Splash 150–250。把握：中。
- E6 重启后命中：Splash 没有（cache 在内存）；lmk 有。把握：高（README 未提；若有，是个新事实）。
- E7 内存：Splash 启动会按 Metal 建议吃满上下文预算；停掉 lmk 后再量。

## 过程与结果
1. 装：`brew install incoai/tap/splash` 几分钟；`splash serve` 自己下 17.4 GB（79 个文件，约 10 分钟）到 `~/Library/Application Support/Splash/models/`
   （另在 HF cache 留 16 GB）。8000 端口被本机别的进程占着，用 `--port 8010`。启动 13 秒到 Ready，"Kernel policy for GPU family 9 with 60 cores"
   （这台是 60 核的 M3 Ultra）。**它拒绝第二个实例**（"already serving"），kill 后要等进程真退出——驱动脚本因此撞了两次。
2. **第一轮（`raw/*-attempt*`、`raw/splash-probe.json`）：探针被 Splash 的流骗了两次**：它开流先发一个 `content: ""` 的 chunk，lmk bench 的"首 token"只看键不看内容，
   于是 4k 冷 prefill 读成 "0.01 s / 576k tok/s"；引擎崩了以后错误流也被当成功。**Splash 自己的服务端日志才是准的**（`raw/splash-serve-2-clean.log`）：

| Splash 服务端日志，第一轮 | 数 |
|---|---|
| 4k 冷 prefill | TTFT 12.6 s（= 320 tok/s） |
| 4k 命中 | TTFT 0.3 s（cached 4,032） |
| decode 400 token，散文 / 代码 | **22.9 / 22.9 tok/s** |
| 32k 冷 | TTFT 108.7 s（= 295 tok/s） |
| 32k 命中 | TTFT 0.3 s（cached 32,032） |
| 4 条短 prompt 并发 | 第一条答完（8 token，8.7 tok/s）后 **"native transport stopped after an engine failure"**，此后每个请求 `runtime_unavailable`，服务不自愈 |
| 重启后再发同一 32k prompt | **cached 0，TTFT 108.9 s**——cache 不跨重启（E6 命中） |

   lmk 同一轮（客户端计时，探针没被骗，`raw/lmk-probe.txt`）：4k 冷 321 tok/s（12.58 s）、命中 0.91 s、散文 44.3 / 代码 58.5（接受 88%）、
   32k 冷 110.8 s → 命中 0.99 s、4 并发合计 59.5 tok/s（`max_parallel: 2`，后两条排队 13.7 s 才出首 token）、**重启后 32k 命中 1.01 s**。
3. **lmk 命中的 0.9 s 花在哪（owner 的猜测：是 256 块的尾巴，不是 SSD）**——第一轮 `lmk-probe.json` 的 `restore_ms` 能拆开：

| 命中 | 磁盘还原 `restore_ms` | 尾巴 prefill（按 321 tok/s） | 其余（HTTP、分词、引擎一步） | 首 token |
|---|---|---|---|---|
| 4k（缓存 3,840，尾巴 194） | 80 ms | ~600 ms | ~230 ms | 911 ms |
| 32k（缓存 32,000，尾巴 34） | 532 ms | ~106 ms | ~350 ms | 987 ms |

   4k 档是尾巴的账（owner 对），32k 档是磁盘的账（约 60k tok/s 的还原）；两档各剩 230–350 ms 固定开销。Splash 整个命中 0.3 s：cache 在 GPU 内存、按 token 粒度
   （4,034 里只有 2 个没命中）。三个可分别做的改进：缓存块尾巴（或缩小块）、内存热层、剖那 0.3 s；都不动"SSD 持久化"这个前提（own-engine.md 09-24 补记）。
4. 探针修了（首 token = 第一个非空文本；错误当结果记；并发分 2 / 4 两档）。**第二轮（`raw/*-r2.*`，客户端计时，一次一个服务）**：

| 同一台 M3 Ultra 96 GB，Qwen3.8-27B 4-bit，temperature 0 | Splash 0.x（brew，2026-09-24） | lmk main（kv 8-bit + MTP 草稿 + thinking off） |
|---|---|---|
| 4k 冷 prefill | 320 tok/s（首 token 12.60 s） | 321 tok/s（12.59 s） |
| 4k 命中首 token | **0.34 s**（cached 4,032 / 4,034） | 0.91 s（cached 3,840：256 块的尾巴 + 磁盘 80 ms） |
| decode 400 token，散文 | **22.9 tok/s** | 44.3 |
| decode 400 token，代码 | **22.9 tok/s** | **58.2**（草稿接受 88%） |
| 32k 冷 prefill | 108.8 s（295 tok/s） | 110.7 s（289 tok/s） |
| 32k 命中首 token | **0.38 s** | 0.98 s（磁盘还原 532 ms） |
| 2 条并发 | **引擎报错** "target policy selected an invalid next anchor"，两条都失败 | 59.7 tok/s 合计（两行普通解码，投机只在单行） |
| 4 条并发 | **503 engine recovering**，全部失败 | 58.9 合计（`max_parallel: 2`，后两条排队 13.8 s） |
| 重启后 32k 命中 | **没有**：cached 0，108.9 s 重算 | **1.07 s**，cache 在磁盘上 |

   结论（这台机器、这个版本）：E1 **落空且反向**——Splash 的 decode 只有 lmk 普通解码的 58%、代码投机的 39%，散文代码一个数，草稿像没生效；
   E2 落空（prefill 持平，两边都是 320）；E3、E4 命中（命中快 2.5–2.7 倍，靠 GPU 内存里的 token 粒度 cache）；E5 落空且反向（并发直接崩）；E6 命中（不跨重启）。
   Splash 的内存规划（`raw/splash-status-r2.json`）：常驻 18.7 GB（权重 15.2 + 草稿 1.27 + 视觉 0.93 + 状态），KV 页 32 token、int8，虚拟预留 62 GB，`maximum_batch_width` 4。
5. 追加测试（`extra.sh` / `req.py`，`raw/extra*.log`，`raw/splash-status-T*.json`）：草稿到底有没有生效（`/status` 的 `metrics.draft_acceptance_rate`）、缺省采样、1000 token 长输出、
   非流式、两条并发的崩溃栈（`transport.last_crash_trace`）、`--max-context 32K`、思考开着（模板缺省档）。
   **结果：Splash 在这台机器上输出的是乱码，前面的所有速度数都是一个坏掉的引擎的数。**
   - 每个回答都是垃圾：代码题 "```odataodataodataadona…"，散文题 "E anon anonadonaadona anon…"；流式、非流式、temperature 0、缺省采样、`--max-context 32K` 全一样。
     （我第一、二轮只看了 usage 和计时，没看文本——探针的第三个 bug，已加 `text[:60]`。）
   - `/status.metrics`：**drafted 15,421 · accepted 0 · rate 0**；ITL p50 43.8 ms = 22.9 tok/s：每步验证一整块（7 草稿 + 1）、接受 0、只出 1 个 token。
     散文代码一个数，因为每步都是"整块全拒"。目标模型自己就在胡说，草稿当然一个不中。
   - 思考开着（模板缺省档，不传 `--default-reasoning-effort`）：**启动预热就失败**，"runtime bootstrap failed [decode_warmup]: target policy selected an invalid next anchor"——
     和两条并发时把引擎打崩的是同一个错。`transport.last_crash_trace` 为空。
   - `--kv-format` 这个 README 写了的 flag 在这版里不存在（"unrecognized arguments"）。
   - 预热自检（`warmup.decode_b1/b2/b4/draft_verify_commit`）全报 True——它的自检没有校验输出内容。
   - 前面表里仍然成立的只有**机制类**的数：命中路径 0.3 s（GPU 内存、token 粒度）、不跨重启、内存规划 18.7 GB 常驻；prefill/decode 的 tok/s 作废。
   **已知问题**：他们的 issue #130（2026-09-24 当天）："[Metal] 1.0.2 decode garbling on Apple GPU family 9 (M3 Ultra): simdgroup K-split reduction on plain
   device partials not coherent"——正是这台机器、这版（brew 装的 1.0.2），根因是 Metal 内核里 simdgroup K-split 归约的一致性问题。他们只在 M5 Pro 上报过数，
   README 写 "Apple M3 or newer"。要看真实性能，等修复版或换 M4 / M5 的 Mac。
