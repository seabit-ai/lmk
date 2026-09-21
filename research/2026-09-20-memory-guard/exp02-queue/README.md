# exp02 — 有了准入队列之后，重做 exp01 的 M4

日期 2026-09-20。对象：m3u 上的 lmk build `753d40a`（准入队列已接入，`max_parallel=2`，tokenBudget=979,877）。
exp01 的 M4：A 在生成，2 秒后 B 带 13k 的全新 prompt 进来 → B 读 prompt 的 43 秒里 A 只出了 11 个 chunk。

## 方法
同 exp01 的 `run.py M4 1 --late-cold`（脚本原样复用，不改）。B 进来约 3 秒后另抓一次 `lmk status`。

## 预期（跑之前写的）
- B 被队列拦住：status 的 waiting 里有 B，原因是"another request is being answered, and this one has 13,4xx tokens of new prompt to read first"。
- A 不受影响：300 个 token 约 8 秒写完，速率约 40 tok/s（exp01 M1：39.7）。
- B 的 TTFT ≈ A 剩下的时间（约 6 秒）+ 读 prompt 的 43 秒 ≈ 49 秒；B 多等的这几秒就是换来的代价。
- 日志里 B 的 `uncachedEstimate` 与 `uncachedActual` 应当接近（全新 prompt：两者都约等于 prompt 长度）。
- 把握 85%。最可能出错的地方：preflight 对 cache 命中的估计不准，让 B 被误判成"短"而放行。

## 结果
原始输出 `raw/results.jsonl`、`raw/status-midway.txt`、`raw/agent-step.jsonl`。

| | exp01 M4（无队列） | exp02（有队列） |
|---|---|---|
| A 写 300 token | 51.3 秒；B 读 prompt 期间只出 11 个 chunk | **8.2 秒，39.4 tok/s**；同一时间窗里出了 238 个 chunk |
| B 的 TTFT | 43.0 秒 | 51.1 秒（多等的 8 秒 = A 写完的时间） |

中途的 status：
```
  busy       1 request
               reading prompt 0 / 78 (0%) · exp01-M4-queued · gen-0 · 4s
  waiting    1 request
               exp01-M4-queued · cold-late · 2s · another request is being answered, and this one has 14,061 tokens of new prompt to read first
```
队列的估计 vs 引擎实测：gen-0 78 = 78；cold-late 14061 = 14061。

### 追加：最不能错的那种情形——agent 的下一步（`agent_step.py`）
A 在写（400 token）；2 秒后来一个 27,190 token 的会话、lmk 已缓存 27,136。**没有被排队**：2.17 秒出字（单独跑约 1–2.5 秒），
A 12.6 秒写完。估计 54 = 实测 54。（载荷取自真实会话，含私人内容，在 nova `2026/2026.0920.M3uOmlxVsLmk/`，不在本 repo。）

### 对照预期
全部 ✅。预期里担心的"preflight 估计不准"没有发生：三次请求估计值与实测值逐个相等。
附带发现（已修）：短 prompt 的请求已在生成，status 仍显示 "reading prompt 0 / 78"——短 prompt 只有一次 begin 回调，
永远等不到 "processed == total"；改为以"答案的第一个 chunk 发出"为进入生成阶段的凭据。
