# exp02：stop 改由 lmk 在回答段匹配之后，同一请求再跑

日期 2026-09-21。m3u，Qwen3.8-27B-MLX-4bit，lmk 分支 `sampling` 提交 f17221a，独占。exp01 的 stop 请求原样重跑（参数没变，
变的是实现——所以是新实验，不覆盖 exp01）。

## 预期（跑之前写）
1. stop=["BANANA"]，prompt 要求先写 BANANA 再写 APPLE：思考里出现 BANANA **不再停**；回答段在 BANANA 之前就停 ⇒
   `content` 为空串或只有 BANANA 之前的字（可能是 ""），`finish_reason=stop`，reasoning 完整。把握：高。
2. stop=["APPLE"]：`content` 为 "BANANA" 加上到 APPLE 之前的分隔（空格/换行），不含 APPLE。把握：高。
3. 非流式与流式结果一致。把握：高。

## 结果（2026-09-21，raw/）
| 请求 | finish | content | 思考里有 BANANA | completion tokens |
|---|---|---|---|---|
| stop=["BANANA"] | stop | 空（回答第一个词就是它） | 是，没停 | 262 |
| stop=["APPLE"] 非流式 | stop | `"BANANA "` | 是 | 193 |
| stop=["APPLE"] 流式 | stop | `"BANANA "` | — | — |

1–3 全部命中。对照 exp01：同一 stop=["BANANA"] 请求，引擎侧匹配在思考第 12 个 token 就停；lmk 侧匹配让思考跑完（262 token）、在回答段截住。
