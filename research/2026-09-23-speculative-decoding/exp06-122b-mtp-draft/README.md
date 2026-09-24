# exp06：Qwen3.5-122B-A10B 自带的 MTP 头做草稿——同一配方能不能复制到 MoE

日期 2026-09-24。m3u（96 GB）。引擎 fork `lmk` 分支 7a1e17f（含 exp05 的量化校验修复）。
起因：owner "draft decoding is very cool, can we add it on other models?" 查到 122B 原版也带 `mtp_num_hidden_layers: 1`，
mlx-vlm 的 `qwen3_5_mtp` 草稿器明确支持 MoE 版（专家权重转换在 `qwen3_5_mtp.py:473-483`），引擎只认这一种草稿器——正好是它。

## 事实（跑之前查的）
- 原版 `Qwen/Qwen3.5-122B-A10B`：48 层，hidden 3072，256 专家 / 每 token 8 个，`moe_intermediate_size` 1024，共享专家 1024。
  MTP 层 785 个张量在第 37–39 分片（18.2 GB 下载）。按配置算 MTP 层 **约 2.5B 参数：bf16 4.7 GiB，4 位约 1.3 GiB；每 token 只动 0.16B**。
- 27B 的草稿器是 810 MB bf16、block_size 3（exp02）。122B 的头大 5.8 倍，但活动参数小。
- 表里两款（`-4bit` 65 GiB 常驻、`-48gb` 44 GiB 常驻）都用同一个头（兼容判据：hidden、词表）。

## 怎么量
1. `split.py --model <本地快照目录> --output ~/.cache/lmk-research/qwen3.5-122b-a10b-mtp-draft`（bf16）。
2. `smoke.py`（exp05 同款，主模型换 `mlx-community/Qwen3.5-122B-A10B-4bit`，`max_seq_nums=2`）：三题贪心无/有草稿（sha 对照）、
   temp 1.0 接受率、双请求。跑之前 `lmk down`（122B 65 GiB + 27B 16.5 GiB + 草稿器 5 GiB 超 96 GB），跑完 `lmk up`。
3. 装得下再看：草稿器要不要量化到 4 位（`-48gb` 在 64 GB Mac 上：44 GiB + 4.7 GiB + 上下文）。

## 预期（跑之前写）
- E1 拆分工具原样吃 MoE 头，产出能被 `load_drafter` 认成 `qwen3_5_mtp`。把握：中高——mlx-vlm 写了 MoE 分支，但我们是第一个在 122B 上用它的。
- E2 贪心 code / copyedit 与无草稿逐字节一致，story 分叉（27B 两次都这样）。把握：中。猜错的一侧：MoE 路由在 block 验证和逐 token 下走不同专家 → 更容易分叉。
- E3 接受率：低于 27B 的 86–87%。MTP 层只有一层 MoE，主模型 48 层；27B 的头对 27B 也是一层对 64 层，但 122B 的 top-8 路由更"跳"。猜 70–80%。把握：低。
- E4 速度：无草稿 60 tok/s（`docs/benchmarks.md`）。草稿一步便宜（0.16B 活动 + lm_head），若接受率 75%：code 约 1.3–1.4×（80 tok/s 上下）。把握：低。
- E5 草稿器 bf16 4.7 GiB 装进 96 GB 的 `-4bit` 没问题；`-48gb` 在 64 GB 上要 4 位草稿器。把握：中高（算的，不是量的）。

## 过程与结果
1. 下载三个分片 17 GB，8.5 分钟（`~15 MB/s`）。`split.py` 吃本地快照目录 3 秒出结果：`model.safetensors` 5.05 GB bf16、`block_size` 3、
   `text_config.model_type: qwen3_5_moe_text`、256 专家（算的 4.7 GiB 对上）。
2. **第一次冒烟：草稿器加载失败**（`raw/smoke-run1-experts-not-stacked.txt`）：`Received 768 parameters not in model: layers.0.mlp.experts.<i>.{gate,up,down}_proj.weight`。
   原版权重是逐专家张量；mlx-vlm 的 MoE 草稿器 `sanitize` 只转 HF 新版的融合布局 `experts.gate_up_proj`，其余情况期望 MLX 的堆叠布局
   `layers.0.mlp.switch_mlp.{gate,up,down}_proj.weight`（形状 `(256, 1024, 3072)` / `(256, 3072, 1024)`，用空模型 `tree_flatten` 读出来的）。
   拆分工具对 MoE 头少做了这一步（27B 是稠密头，没这个问题）。**E1 猜错的一侧**：工具吃得下 MoE 头，但产出的布局加载不了。
   修：`restack_experts.py` 把 256 个逐专家张量按序 `mx.stack` 成三个张量，值不变，原地重写 safetensors（20 个键），`load_drafter` 认出 `mtp`。
   发布到 HF 的就是重排后的文件（原版 mlx-vlm 直接可载）；主模型普通解码三题 60.3–60.4 tok/s（与 benchmarks 一致）。
3. **第二次冒烟（`raw/smoke.txt`，重排后的草稿器）：跑通了，但结果是负的。**

| 贪心 400 token | story | code | copyedit |
|---|---|---|---|
| 普通解码 | 60.1 tok/s，6519c07c4b | 59.8，594652d587 | 59.8，8fe1ef2be6 |
| 有草稿 | 31.0（**0.52×**），5638dfbf94 **分叉**（374 token） | 30.8（0.52×），f5ada74112 **分叉** | 30.9，一致 |

   接受 **0 / 2555**（1278 轮）；temp 1.0 也是 0 / 797。双请求（投机不开）42.9 / 41.9，合计 78.1。
   两个嫌疑，得分开查：
   - **草稿器本身没用**：0% 远低于随机——权重映射错（专家堆叠次序、`mlp.gate` 路由、共享专家门控）或 MTP MoE 层的前向在 mlx-vlm 草稿器里本就没验证过。
   - **MoE 的校验前向与逐 token 不一致**：drafts 全拒时每轮只吐 bonus token，那应该逐字节等于普通解码，可 code 分叉了、story 少了 26 个 token。
     27B（稠密）上同样的路径是逐字节一致的（exp03/05）。嫌疑点：`target_verify` 分支下的 MoE `switch_mlp`（gather_qmm）或 GDN 的多 token 路径。
   下一步：绕开引擎，直接用 mlx-vlm 自己的循环（exp02 的 `generate_step` 路径，`draft_kind="mtp"`）在 122B 上跑一遍：
   若它也是 0%，是草稿器/上游 MoE 支持的问题；若它正常，是引擎 wiring 对 MoE 的问题。
   **E2/E3/E4 全部落空**；猜错的一侧不在"接受率高低"，而在"能不能用"——我把"mlx-vlm 写了 MoE 分支"当成了"MoE 分支验证过"。
