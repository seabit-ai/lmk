# exp01：stop 串碰到思考段；temperature 0 与缺省

日期 2026-09-21。m3u，Qwen3.8-27B-MLX-4bit，lmk 分支 `sampling`，独占，单请求。

## 预期（跑之前写）
1. **stop 在思考里**：prompt 让模型输出 BANANA，stop=["BANANA"]。引擎的 stop 处理器对整条 token 流生效、不知道思考
   和回答的区别，所以**思考段里第一次出现 BANANA 就停**：`reasoning_content` 被截断，`content` 为空，`finish_reason=stop`。
   把握：中等偏高（引擎源码 `create_stop_string_processor` 装在批处理生成循环上，位置在 lmk 的切分器之前）。
2. **stop 串本身不进输出**（OpenAI 语义）。把握：中等——引擎的 processor 有 "stop string 已匹配" 的分支，是否回吐前面的
   部分 token 没读透。
3. **temp 0 两次输出逐字相同；缺省（temp 1.0 / top_p 0.95 / top_k 20）两次不同**。把握：高。

## 怎么跑
`run.sh`：三组请求，原始响应落 `raw/`。

## 结果（2026-09-21，raw/*.json）
| 请求 | finish | content | reasoning 结尾 | completion tokens |
|---|---|---|---|---|
| stop=["BANANA"] | stop | 空 | `…"Write the word ` | 12 |
| 同 prompt 无 stop | length | 空（400 token 全在思考里，模型对这个小任务想过头，与本实验无关） | — | 400 |
| temp 0 ×2 | stop | 逐字相同（"Red is the warm, urgent glow…"） | 相同 | 122 / 122 |
| 缺省 ×2 | stop | 不同（Blue… / Red…） | 不同 | 192 / 121 |

1. ✓ 预期 1：stop 串在**思考段**第 12 个 token 处命中即停，回答为空。引擎的 stop 处理器对整条流生效。
2. ✓ 预期 2：停止串本身不在输出里（reasoning 止于 BANANA 之前）。
3. ✓ 预期 3：temp 0 可复现；缺省采样两次不同。

结论进 notes.md SMP-002：把 stop 交给引擎会毁掉思考型模型的回答 ⇒ lmk 自己在回答段匹配。
