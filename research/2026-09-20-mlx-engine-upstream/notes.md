# 向上游 mlx-engine 提持久化前缀 cache

日期 2026-09-20。owner："the mlx-engine feature request is worthy doing. can you go do it?"
mlx-engine 是 **LM Studio 公司**开源（MIT）并维护的引擎（`github.com/lmstudio-ai/mlx-engine`）；lmk 的
持久化 cache 现在靠替换它内部的一个类名实现（lmk 设计 §6.6），上游接受注入点才算稳。编号前缀 `UPS-`。

## 发现

### UPS-001 上游的规矩：功能请求先开 issue；不欢迎没打招呼的功能 PR
`CONTRIBUTING.md`："The best way to communicate with the team is to open an issue"；"If you find an existing
issue you'd like to work on, please comment on it first and tag the team"；"We discourage drive-by feature PRs
without prior discussion"。首次贡献要签 CLA（bot 在 PR 上提示）。

### UPS-002 已有的相关 issue（2026-09-20 抓取）
- **#354 "Disk prefix KV caching"**（open，2026-07-24，richardwerkman）——要的就是我们做的：按模型指纹把
  KV cache 持久化到磁盘、跨卸载与重启命中、可配位置与容量、LRU 清理。**无任何维护者回应**。提的人担心
  "needs to be wired into the actual frontend which isn't open source"。
- #335 "custom disk KV cache path"（closed，2026-06-11，关闭原因页面上看不到）。
- #341 "Add configuration toggle to disable prompt disk caching"（open）。
- **#366 "MLX auto-fit replaces the configured context length and disagrees with the prompt cache budget"**
  （open，2026-08-15）——我们也撞上了：lmk 配的是 200,000，引擎日志
  `configured=200,000 fitted=262,144 effective=262,144`，磁盘预算 `cap_gib=162.81`。

### UPS-003 做法：在 #354 下评论，不另开重复的 issue
评论原文：`comment-issue-354.md`（英文，写给上游维护者与 #354 的作者）。内容：实测数据（冷启动 36.8s →
重启后 2.4s）、我们在引擎外面不得不怎么做、持久化实际需要的四样东西、对"要接进闭源前端"这个担心的回答
（不需要）、一个拆成两步的小 PR 提议（先注入点，再持久化存储）、两个实现时该定的事（显式容量上限、
身份键带格式版本）。没有提 lmk / kitten 的名字——它们还没公开，没有可给的链接。

### UPS-004 发不出去：我这边的 shell 没有可用的 GitHub 凭证
`gh auth status`：bruce-claw 与 owner 两个账号的 token 在我的 shell 里都读不到（钥匙串够不着，
与记忆 m3u-git-identity 记的一致）。按约定不绕凭证——评论由 owner 在自己的终端发：

    gh issue comment 354 --repo lmstudio-ai/mlx-engine --body-file research/2026-09-20-mlx-engine-upstream/comment-issue-354.md

## 决策日志
- **评论 #354，不新开 issue。** 理由：UPS-001 的规矩 + #354 要的就是这个功能；新开一个会被当重复关掉。
- **现在不提 PR。** 理由：上游明说不欢迎没讨论过的功能 PR；#354 两个月没人回，先看维护者的态度。
  代码是现成的（`lmk/lmk/persistcache.py`），他们点头就能改成上游的形状。
- **否决：直接 fork 一份引擎自己维护。** 理由：现在钉在 `08f0c07` + 替换类名已经够用；fork 意味着以后
  自己跟新模型架构，代价远大于一个脆弱的注入点。

## 未定
- 上游不回应怎么办（#354 已晾了两个月）。届时的选项：继续钉 commit + 替换类名；或带着补丁 fork。
