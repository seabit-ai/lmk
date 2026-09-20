# exp01 — uv 能不能一次装好 lmk 的环境

日期 2026-09-20。为 lmk OOBE 设计点 D（docs/design/2026-09-20-lmk-oobe.md）验证前提：安装脚本底下用 uv。

## 方法
全部在会话 scratchpad 里，不碰系统：uv 独立二进制装到 `<scratch>/uvtest/bin`（`UV_NO_MODIFY_PATH=1`），
`UV_PYTHON_INSTALL_DIR`、`UV_CACHE_DIR` 都指向 scratchpad。步骤各自计时：
1. 下载 uv
2. `uv venv --python 3.11`（机器上的 python 不许用：`--managed-python`）
3. `uv pip install -r lmk/requirements.txt`（61 行，含 3 个 git 依赖）
4. 用这个环境 `import mlx_engine, lmk.server`，再跑 lmk 的单测（42 个）

## 预期（跑之前写的）
- 四步全过。把握 80%。最可能的失败点：某个 git 依赖需要本地编译，或 uv 的解析器比 pip 严格、在钉死的版本上报冲突。
- 步骤 2 < 30s；步骤 3 在 2–6 分钟（pip 那次约 5 分钟，uv 应更快）。
- 单测 42 个全过。

## 结果
原始输出 `run.log`。uv 0.12.17，Python 3.11.16（uv 自己下载的），m3u 的网络。

| 步骤 | 结果 | 耗时 |
|---|---|---|
| 1 下载 uv | ✅ | 2s |
| 2 `uv venv --managed-python --python 3.11` | ✅ | 2s |
| 3 `uv pip install -r requirements.txt`（61 行，3 个 git 依赖） | ✅ | **16s** |
| 4 import + 单测 | ✅ 42 passed, 5 skipped（itest 门控） | 8s |

占盘：venv 1.0G，python 77M，uv 下载缓存 1.1G（装完可删）。

### 对照预期
- 四步全过 ✅。没有本地编译，没有解析冲突。
- 步骤 3 我预期 2–6 分钟，实际 16 秒——**低估了 uv 一个数量级**。pip 那次的"几分钟"大头是串行下载和解析。
- 结论：从一台没有 python 的机器到 lmk 环境可用，**约 20 秒**（不含 15G 模型）。安装脚本 + uv 这条路的前提成立。
- 未测：慢网络；没有 git 的机器（3 个 git 依赖要 git；macOS 上首次调 git 会弹 Xcode CLT 安装框——这是一个真前置，见 notes OOBE-003）。
