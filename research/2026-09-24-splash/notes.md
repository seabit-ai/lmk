# Splash（Inco AI）——同定位的对手，2026-09-24 owner 递来的链接

来源：https://inco.ai/blog/splash/（2026-09-17）、https://github.com/incoai/splash。**以下全是对方的主张，未在我们的机器上验证**；
他们的数是 M5 Pro 48 GB 上量的，我们的是 M3 Ultra 96 GB，不能横着比。验证方法同 SVY（`research/2026-09-20-local-server-survey`）：
装在同一台机器，同样的探针（`lmk bench` 的四个数 + 真实 agent 会话的每步出字时间）。

- **SPL-001 定位与 lmk 几乎重合**：Apple Silicon、一次一个模型（"the engine is built around the model"）、brew 一装、OpenAI 兼容
  （另有 Anthropic Messages 与 Responses）、没有配置文件（全是 `splash serve` 的 flag）、默认 127.0.0.1:8000。
  差别：他们的模型是**自己打包的**（`incoai/Qwen3.8-27B-Splash` 17.4 GB、`incoai/Qwen3.6-35B-A3B-Splash` 20.9 GB），"Plain MLX or
  Transformers checkpoints do not work"；只有两款模型；要 macOS 26.4+、M3+、36 GB 起（推荐 48）。owner 的机器：macOS 26.6.2、M3 Ultra，满足。
- **SPL-002 投机解码是唯一解码路径**，草稿器是每模型专训的 **DFlash 2**（27B 的草稿 1.2 GiB），声称接受率高于模型自带的 MTP 头。
  我们：MTP 头（27B 810 MB bf16），code 接受 86–88%、prose 53%；exp05/06。SPD-008 里记过 `z-lab/Qwen3.8-27B-DFlash2` 存在、mlx-vlm 只实现了 v1。
- **SPL-003 不是 MLX**：自写 Metal 内核、按模型维度自动生成 4-bit matmul 与注意力内核（prefill 大批 / decode 小块各一套）；
  启动时按 Metal 建议的内存算出上下文、KV 容量、批上限（"memory plan computed for this machine"）。
- **SPL-004 cache**：按前缀分页的 KV cache，**在 GPU 内存里**（`--kv-format int8` 缺省 / bf16）；混合模型在前缀边界快照 GDN 状态。
  README 没说能跨重启——这是 lmk 的差异点（磁盘持久化，重启后仍 ~1 s 接上）。
- **SPL-005 他们报的数（M5 Pro 48 GB，Qwen3.8-27B）**：短 prompt decode **74 tok/s**（oMLX 38）；32K 上下文 54（oMLX 28）；
  32K prompt prefill 360 tok/s（oMLX 110）；32K 命中首 token **282 ms**（oMLX 2,049）；4 并发合计 170 tok/s（oMLX 43）；16 条 32K 并发能跑。
  35B-A3B：210 / 143 tok/s，prefill 2,000，命中 123 ms，4 并发 357。
  我们（M3 Ultra 96 GB，27B-4bit，docs/benchmarks.md）：decode 39.5 普通 / 58.6 code 投机（prose 44.5）；prefill 321–323；4k 命中首 token 0.91–1.02 s
  （磁盘还原到最后一个 256 块 + HTTP）；2 并发短 prompt 1.7×。**条件不同，先别下结论**——但"32K 命中 282 ms"对"4k 命中 0.9 s"这一格值得单独量。
- **SPL-006 要验证的顺序**：① 同机装 splash，跑 `lmk bench` 同款探针（冷 prefill 4k、命中、decode 散文/代码）；② 真实 agent 会话（56 工具 11k 系统提示、
  27k 对话）的每步出字；③ 重启后命中；④ 并发 2/4。每条两边同条件，一表。
- **SPL-007 同机实测（exp01，M3 Ultra 60 核，macOS 26.6.2，brew 装的 Splash）：引擎在这台机器上输出乱码**，每个回答都是 "odataodataadona…" 这类垃圾，
  草稿接受 0 / 15,421，decode 固定 22.9 tok/s（每步 43.8 ms 验证整块、全拒、出 1 个 token）；思考开着时启动预热失败、两条并发把引擎打崩
  （同一个错 "target policy selected an invalid next anchor"），崩后所有请求 `runtime_unavailable`、不自愈。他们的自检全报 True 但不校验内容。
  他们的 issue #130（同日）确认："1.0.2 decode garbling on Apple GPU family 9 (M3 Ultra): simdgroup K-split reduction … not coherent"。
  ⇒ **他们博客的数在这台机器上无法复现，性能比较作废**；能比的只有机制：命中 0.34 s（4k）/ 0.38 s（32k）对 lmk 0.91 / 0.98，cache 不跨重启，prefill 两边都是 320 tok/s（他们算的还是垃圾）。
- **SPL-008 探针的教训**：三个 bug 全是"没看内容"——空 chunk 骗首 token、错误流当成功、乱码当回答。以后对别人的服务器至少打印回答的前 60 字，
  `lmk bench` 也该把 decode 探针的回答头几个字印出来（一个坏掉的引擎会把 tok/s 量得很好看）。
