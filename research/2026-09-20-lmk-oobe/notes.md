# lmk 开箱体验（OOBE）：取证

日期 2026-09-20。为 `docs/design/2026-09-20-lmk-oobe.md` 的 SAD 供事实。编号前缀 `OOBE-`。

## 发现

### OOBE-001 今天装 lmk 要八步，含三个不该由用户做的决定、一个隐藏前置、一个对竞品的依赖
逐步走查见设计文档 §1。要点：端口/窗口/模型路径三个必填；`PYTHON311 ?= ~/.local/bin/python3.11` 是 m3u 私有路径；
README 不提模型从哪来，示例路径指向 LM Studio 的目录；没有"它起来了吗"的命令；第一句冷 prefill 37s 无提示。

### OOBE-002 HF 共享目录是 `~/.cache/huggingface/hub/`；m3u 上同一批模型散在三处
`HF_HOME` 可改；布局 `models--<org>--<name>/{blobs,refs,snapshots/<commit>}`，commit hash 可直接当模型身份。
m3u：`~/.lmstudio/models` 166G、`~/models`（oMLX）32G、HF cache 16G。Qwen3.8-27B-4bit 的实体在 LM Studio 目录，HF cache 里是 4K 空壳。

### OOBE-003 uv 能在约 20 秒内从零装好 lmk 的环境（exp01）
uv 2s + 自带下载 python 3.11 2s + 61 行 requirements（含 3 个钉 commit 的 git 依赖）16s；42 个单测全过。
mlx-engine 本身不是包（无 pyproject/setup.py），仍需按 commit 取源码放上 PYTHONPATH——tarball 即可，不必 git clone。
**剩下的真前置是 git**：3 个 git 依赖需要它；全新 macOS 首次调 git 会弹 Xcode Command Line Tools 安装框。未测绕法
（把 git 依赖换成 GitHub 的 commit tarball URL：`pkg @ https://github.com/…/archive/<commit>.tar.gz`）。

### OOBE-004 prefix cache 现状：10G / 一天半，无 lmk 侧上限
`~/.kitten/lmk/cache/<模型身份>/`，209 条记录；09-20 当天新增 114 个文件，约 5G/天；引擎预算 162.81 GiB。
同机 oMLX 的 cache 无人过问涨到 176G（上限设的 185GB）。Time Machine 视角该目录为 Included。

## 未测
- 慢网络下的安装时长；无 git 机器；tarball 形式的 git 依赖。
- `hf download` 15G 的实际耗时与断点续传表现。
