# Qwen3.8-27B-4bit 对 Qwen3.5-122B-A10B-4bit：哪个更聪明

日期 2026-09-23。起因：owner "which one has more intelligence?"。09-19 那场评测（LMS-007）站不住的四点这次全避开：
机器判分、每题多次、存原始输出、思考作为变量控制。

## 设计
**四臂**（每臂一个临时 lmk，端口 1236，模型的推荐配置）：
| 臂 | 模型 | 思考 |
|---|---|---|
| 27B-xhigh | qwen3.8-27b-4bit | 缺省（reasoning_effort xhigh） |
| 27B-low | qwen3.8-27b-4bit | reasoning_effort low |
| 27B-off | qwen3.8-27b-4bit | thinking: false |
| 122B-off | qwen3.5-122b-a10b-4bit | thinking: false（它的推荐配置；开着会数数循环，MDL-009） |

**四类题，全部机器判分，各出通过率，不合成总分：**
| 类 | 来源 | 题数 | 判分 |
|---|---|---|---|
| math | GSM8K test（openai/grade-school-math），固定种子抽 50 | 50 | 末行 `Answer: N` 与参考数值相等 |
| code | HumanEval（openai/human-eval），固定种子抽 40 | 40 | 提取 ```python 块，跑题目自带 check() 测试，10 s 超时 |
| instruct | 自写 10 条格式约束（恰好三条要点、合法 JSON、全大写、不含字母 e……） | 10 | 程序核对约束 |
| tools | 自写 5 个多步工具任务，harness 里的内存文件系统 | 5 | 终态或最终答案核对 |

每题 **3 次**，模型缺省采样（temp 1.0 / top_p 0.95 / top_k 20），max_tokens 8000；`finish=length` 算错并单独计数。
记录每次的 completion tokens 与耗时——"多聪明"和"多贵"并列报。并发 2（lmk 的 max_parallel）。

## 预期（跑之前写）
1. **math**：122B-off > 27B-off；27B-xhigh ≥ 122B-off（思考把 27B 抬上去），但每题耗时 3–5 倍。27B-low 介于中间。把握：中。
2. **code**：同 1 的排序。把握：中低——HumanEval 上 4bit 量化的 27B 可能撞上限。
3. **instruct**：27B-xhigh 最好（思考帮它核对格式），122B-off 次之。把握：中。
4. **tools**：四臂都接近满分（今天的小任务都全对），差别在 token 数：27B-xhigh 每步几百到几千，其余几十。把握：中高。
5. **length 超限**：27B-xhigh 会有几题思考撞 8000；122B-off 零。把握：高。
6. 不预设总胜负；每类各说各的。
