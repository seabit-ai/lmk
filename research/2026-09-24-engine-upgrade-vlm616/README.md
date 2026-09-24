# 引擎升级：mlx-vlm 0.6.12 → 0.6.16（为 DFlash 2 铺路；own-engine.md 09-24 补记）

日期 2026-09-24。m3u（96 GB）。fork 分支 `lmk-vlm616`（从 `lmk` e1239e1 起）；lmk 分支 `engine-vlm616`。上游 mlx-engine 至今仍钉 mlx-vlm 321514d（0.6.12），
所以升级是我们自己的：fork 的 `requirements.txt` 换 pin（2b31570b = v0.6.16）+ 新依赖 `mlx-audio==0.5.6`；`mlx==0.32.0`、`transformers==5.15.0` 都够。

## 0.6.16 改了什么、我们的补丁怎么跟
- `qwen3_5/language.py` 改了 873 行：`Qwen3_5Attention.__call__` / `Qwen3_5GatedDeltaNet.__call__` 去掉了 `target_verify` / `gdn_sink` 参数；
  校验块改由 `LanguageModel.__call__(speculative_verify=True)` 交给 `Qwen3_5ExactSpeculativeVerifier`（`speculative_verifier.py`，一套按 T 行输入
  定制的量化 matmul / argmax Metal 内核 + 自己的 attention/GDN 实现）。**普通前向不再返回 `gdn_states`**，回滚需要它——所以 exact verifier 是唯一的校验路径。
- 引擎自带补丁（`patches/qwen3_5.py`）：attention / GDN 的补丁去掉那两个参数（快路径条件不变）；语言模型补丁不用改（`capture_layer_ids` 非 None 就走原路）。
- 我们的量化校验钩子（7a1e17f 挂在 `_target_verify_left_padded_attention`）：函数没了；bug 还在——verifier 的 `_attention` 先问
  `language._qwen3_5_left_padded_attention`（量化 cache 一律 None），再按稠密 keys 逐位置切片，`'tuple' object has no attribute 'shape'` 原样复现。
  它是**运行时按模块属性**查这个 helper 的，钩到新名字即可（cb476a1）。
- `speculative.py` 的 `_verify_block` 加 `speculative_verify=True`。mtp.py 的四个 helper 签名 0.6.12 → 0.6.16 没变。
- 引擎全套单测：失败集与基线 60 个完全一致（要模型 / 网络 / stdin 的那些）。

## 验收（GPU）
| | 0.6.12（exp03 / exp05） | 0.6.16 |
|---|---|---|
| `make cache-compat`（旧 cache 新引擎能还原） | — | **过**，`CACHE_FORMAT_VERSION` 不动 |
| itest kv8 + 草稿 | 5/5 | **5/5** |
| kv16 + 草稿 贪心：story / code / copyedit | 分叉 / 一致 / 一致 | **一致 / 一致 / 一致** |
| kv16 + 草稿 倍数 | 1.14× / 1.49× / 1.48× | **0.92× / 1.26× / 1.24×** |
| kv8 + 草稿 贪心 | 分叉 / 一致 / 一致 | 分叉 / **分叉** / 一致 |
| kv8 + 草稿 倍数 | 1.14× / 1.49× / 1.48× | 0.96× / 1.23× / 1.29× |
| 采样 temp 1 接受率（code） | 86–87% | 85% |
| 普通解码 | 39.5 | 39.2（不变） |

接受率和轮数几乎不变（598 轮 / 61% 对 582 / 64%），**每轮变慢了约 15–20%**——账在校验前向，不在草稿器。
试过"不走 exact verifier"：普通前向不返回 `gdn_states`，回滚直接崩，此路不通（开关已删）。

## 微基准（scratch `verify_bench.py` / `verify_bench2.py`，27B-4bit，单序列，暖机后 20 次平均）：一次 4 token 校验前向

| | ms | 对 (a) 的最大 |hidden| 差 |
|---|---|---|
| 普通 1 token 解码步 | 28.3 | — |
| 普通前向喂 4 token（只取 hidden；不能用：没有 `gdn_states`） | 43.3 | — |
| (a) exact verifier 原样 | 59.2–60.0 | 0 |
| (b) exact verifier，T 行融合 matmul 换成普通模块调用 | 50.8 | **0.5** |
| (c) (b) + 注意力逐位置循环换成整块一次 | 49.8 | 0.5 |
| (d) exact 融合 matmul + 整块注意力 | 58.3 | **0** |

结论：回退的 17 ms 里约 8 ms 是 exact verifier 的定制 T 行 matmul（它买到的正是"与普通解码逐字节一致"——换成普通 matmul，hidden 差 0.5），
约 1 ms 是逐位置注意力（换成整块一次不改数值，但省不了什么），剩下约 8 ms 在 verifier 的其余部分（GDN 校验实现等，没再拆）。
kv8 + 草稿的 code 分叉：我的量化整块注意力在 L=4 与普通解码的 L=1 走的是同一个 `quantized_scaled_dot_product_attention` 但形状不同，
不保证位级一致；0.6.12 上 code 恰好一致是运气。kv8 本来就"答案略有不同"（模型页 Known issues），接受。

## 裁决（owner 2026-09-24：①）
两个选项：① 照 0.6.16 的 exact verifier 用：27B MTP 头 code 1.49× → 1.26×（M3 Ultra），换来三题逐字节一致——"开不开投机答案一样"的承诺完整成立；
DFlash2 接进来走的也是它。② 钩回普通 matmul：约 1.35×，但一致性退回 0.6.12 的水平甚至更差。我推荐 ①，**owner 裁 ①**。
