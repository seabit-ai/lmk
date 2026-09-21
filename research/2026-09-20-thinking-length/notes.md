# 一步想了 8 分钟：取证

日期 2026-09-20。起因：owner 盯着 `lmk status -w` 跑 agent，看到一个请求 `decode thinking` 持续 6 分半、9,060 token 还没完
（64,619 prompt、51,200 命中；27 tok/s）。编号前缀 `THK-`。只记结构与数字，不记内容（内容是 owner 的私人邮件分拣）。

## 发现

### THK-001 lmk 从第一版起一直在贪心解码
引擎 `mlx_engine/utils/sampling.py:93-100`：`temp = 0.0 if temp is None else temp`；`if temp == 0: return argmax`。
lmk 忽略请求里的采样参数、也不传缺省值 ⇒ 每一步取概率最大的 token。
模型随权重发布的 `generation_config.json` 写的却是 `do_sample: true, temperature: 1.0, top_k: 20, top_p: 0.95`。
贪心解码对思考型模型的已知风险是重复打转；**这一次没有发生**（见 THK-003），它对思考长度的影响没量过。

### THK-002 lmk 从不设思考力度，于是每个请求都跑在模型的最高档
模型的 chat template：`{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}`，支持 `xhigh`（缺省）/ `medium` / `low`，
另有 `enable_thinking` 可整个关掉。lmk 渲染模板时两个都不传 ⇒ 永远 `xhigh`。
"力度是 server 级常量、放配置里"早已裁决（见 `docs/backlog.md`"已裁但还没做的"），还没做。

### THK-003 这次的长思考不是打转，是真在干活
从 agent 一侧的 wire dump 取出该请求的 reasoning 流（采样时 26,732 字符、317 句）：**逐句完全重复 2 句（0%）**。
最后几段的开头是："Let me reconsider the scope one more time" / "Actually, can … create nested folders …?" /
"Let me carefully write out the handle lists." / 一段 128 个条目的清单 / "Let me count: Row 1: …"。
即：最高档思考 + 一个诱导枚举的任务（批量处理 128 个对象）——模型在思考里把整份清单手写一遍并逐行点数，之后在工具调用的参数里
还得再写一遍。采样截止时已 10,940 token、7 分 44 秒，仍未结束。

## 判断（不是事实）
- 直接的杠杆是 THK-002：把力度降到 `medium` / `low`。代价：它渲染在 prompt 最前面，改一次 = 全部会话的 cache 冷一次。
- THK-001 应当修（跟随模型作者的采样设置是更诚实的缺省），但别指望它解决思考过长——没有证据。
- agent 一侧也有份：批量操作让模型枚举 128 个 id。那是 agent 的工具设计问题，不是本 repo 的。

## 未测
- 同一请求在 `medium` / `low` 下的思考 token 数与结果质量。
- 采样（temperature 1.0 / top_k 20 / top_p 0.95）对思考长度与质量的影响。
