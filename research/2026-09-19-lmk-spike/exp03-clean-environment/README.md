# exp03 — 干净环境：只用公开仓库和 pip，不借 LM Studio 的任何东西

日期 2026-09-19。lmk 设计 §3 的后果：最终不要 LM Studio ⇒ lmk 不能长期借它自带的 Python 环境。

## 目的
exp01 / exp02 借的是 LM Studio 自带的解释器和库。这里回答：**从公开的 mlx-engine 仓库 + pip 装出来的
环境，是不是同一个东西**——有没有磁盘前缀 cache、命中规则一样不一样、速度持平不持平。
如果公开仓库落后一截或缺那块，lmk 的整个前提要重新评估。

## 方法
- `setup.sh`：在仓库之外的临时目录里 `git clone` 公开的 `lmstudio-ai/mlx-engine`（main），用本机的
  python3.11 建 venv，`pip install -r requirements.txt`（该文件把依赖全钉了版本：`mlx==0.32.0`，
  mlx-lm / mlx-vlm / outlines 钉在具体 commit）。记下引擎的 commit 和 `pip freeze`。
- `run.sh`：**exp01 的同一份 `spike.py`，一个字不改**；`PYTHONPATH` 指向克隆下来的仓库根，解释器用
  venv 的。模型还是磁盘上那份 27B-4bit 权重（权重不属于 LM Studio 的环境）。
- 三个请求同 exp01：A-cold → B-fork → A-again。

## 预期（运行前写下）
1. 装得起来。把握中：依赖全钉死是好事，但 `torchvision` 会拖进 torch（大），git 依赖要现编/现拉；
   python 3.11 与 LM Studio 自带的一致。
2. `load_model` 返回 `BatchedVisionModelKit`，stderr 里出现 `VLM prompt cache` 字样——公开仓库的
   main 有 `prompt_cache/` 目录（LMS-011 对过文件清单）。把握中高。
3. 数字与 exp01 一致：A-cold cached=0、约 320 tok/s；B-fork cached=2560；A-again cached=2560、
   首 token < 1 秒。把握中——`mlx==0.32.0` 与本机那份同版本；但 mlx-lm / mlx-vlm 的 commit 可能比
   LM Studio 打包的那份（0.31.3 / 0.6.5）新或旧，聊天模板渲染出的 token 数可能差一两个。
4. 若第 2 条不成立（公开版没有磁盘 cache），这就是今天最重要的坏消息。

## 对比基准
exp01-engine-standalone（同脚本，借来的环境）。

## 运行
    LMK_SPIKE_DIR=<仓库外的临时目录> ./setup.sh && LMK_SPIKE_DIR=<同上> ./run.sh

## 结果（2026-09-19，`result.jsonl` / `stderr.log` / `engine-commit.txt` / `pip-freeze.txt` / `setup.log` 保留）
环境：mlx-engine main @ `08f0c07`；`mlx==0.32.0`、`mlx-lm @ 2c008fd`、`mlx-vlm @ 321514d`、
`torch==2.9.0`、`transformers==5.15.0`；venv 共 1.2G；python 3.11。

| 步 | prompt | cached | 首 token（干净环境） | 首 token（exp01，借来的环境） |
|---|---|---|---|---|
| A-cold | 2680 | 0 | 8.65s（≈ 310 tok/s） | 8.38s |
| B-fork | 2680 | 2560 | 0.57s | 0.57s |
| A-again | 2680 | 2560 | 0.59s | 0.57s |

对照预期：
1. ✓ 装得起来，一次过（几分钟，主要在 torch）。
2. ✓ `BatchedVisionModelKit`；stderr 里有 `VLM prompt cache disk usage / disk budget / Prompt cache restore`。
3. ✓ 数字与 exp01 **逐项一致**：prompt 同为 2680 token（聊天模板渲染没有差出一个 token）、
   命中 0 / 2560 / 2560、update 同为 [2048, 2560]。冷 prefill 8.65s vs 8.38s，差 3%，在单次测量的
   噪声之内（两次都只跑了一遍）。
4. 不适用——坏消息没有发生。

## 读数
- **公开仓库 + pip 装出来的，就是 LM Studio 在用的那个引擎**（至少在这条路径上）：磁盘前缀 cache、
  命中规则、速度都对得上。lmk 可以完全不依赖 LM Studio 的安装。
- 依赖被上游钉死在具体版本 / commit 上——lmk 可以直接沿用这份钉法，升级是一个显式的动作。
- 1.2G 的环境里 torch 占了大头，而它只是 `torchvision` 拖进来的（mlx-vlm 的图像预处理用）；
  第一版不做图片的话也许能省，没试。
