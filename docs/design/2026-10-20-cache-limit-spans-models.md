# `cache.max_size` 是全局上限：淘汰跨模型，状态行合并报告

日期 2026-10-20。状态：owner 2026-10-20 "agree, go"，已实现（分支 `cache-spans-models`），未合。回到 OOBE §C2 的原意："整目录按最久未用淘汰——一条规则"；实现漏了运行中。

## 现象（owner 的机器，换模型之后）
`cache.max_size: 200G`，目录里旧模型占 ~199.9 GB。换成另一个模型后 `lmk status` 的 cache 行是 `0 B of 77.7 MB`：
200G 减旧模型占的，剩下的才是新模型的上限。owner 的判断：这是 bug——200G 是所有模型共用的上限，最久没用的记录该被淘汰，
不论它属于哪个模型。

## 事实（读代码，未实测；`lmk/persistcache.py`、`lmk/engine.py`）
- 跨模型的"最久没用的先丢"只在启动时做一次：`prepare_cache_root` 在超限时整目录删别的模型的 cache，并返回 `max_bytes - others_bytes`，
  作为当前模型存储的**固定**上限（`engine._install_persistent_cache` 是它唯一的调用点）。
- 运行中的淘汰是存储自己做的，只看自己的 `_total_bytes`，碰不到别的模型的目录。结果：换模型后到下次重启前，新模型最多写 `max_size - 旧模型占用`。
- "最久没用" = 记录文件的 mtime，用到时 `touch`（`PersistentBlobStore.keys_oldest_first` / `touch`），所以各模型目录之间可以直接按 mtime 比。
- `lmk status` 的 cache 行只报当前模型子目录的 used / max：用户配的 200G 在那一行里看不到。

## 裁决（owner，2026-10-20）
1. 这是 bug。运行中，当前模型的存储需要淘汰时，**按记录**在所有模型的记录里选 mtime 最旧的那一条淘汰（别的模型的也算），
   直到整个目录回到 `max_size` 以内。
2. 不按整个模型目录删（我的提议，被否）。启动时 `prepare_cache_root` 的整目录删除是否也改成按记录，随这条一起看。
3. 别的模型的链被抽掉几条之后下次恢复出什么，是**另一个问题**（owner：与这件事无关），不在本设计里。

## 状态行（待 owner 确认格式）
提议：整个目录的数放第一位，当前模型与别的模型拆开：
`cache      199.9 GB of 200.0 GB in ~/.lmk/cache (this model 0 B · other models 199.9 GB) · since start …`
没有别的模型的 cache 时括号省略。只报数字，不写原因。

## 要测什么
- 单测：两个目录共用一个 `max_size`，当前模型写满时先淘汰别的目录里 mtime 更旧的记录；当前模型自己的更旧时淘汰自己的；
  总量始终 ≤ `max_size`；别的目录淘汰光之后才轮到当前模型自己。
- 真机：复现今天的情形（旧模型占满、换模型），新模型的 cache 行上限不再卡在剩余量。

## 实现与验证（2026-10-20）
- `lmk/persistcache.py`：`OtherModelsRecords`（启动时扫一次别的模型目录：总字节、可淘汰的记录按 mtime 排序；一个模型的最后一条记录没了，
  连它的布局文件和目录一起删）、`trim_to_limit`（"谁最旧谁走"，同龄时别的模型的先走）。存储类的 `_evict_if_needed` 改为对整个目录的
  上限做 `trim_to_limit`；上限仍是 `min(max_size, 已用 + 磁盘空闲 - 10 GB)`，只是"已用"含别的模型。`prepare_cache_root` 不再整目录删，
  只建目录、打标记（启动时超限也由第一次存盘的淘汰按记录处理）。
- 日志 `LmkCacheOtherModelsTrimmed`（条数、MiB、清空了几个模型、上限与别的模型剩余）；`LmkChatDone` 多 `cacheOtherModelsMib`。
  `LmkCacheOrphanDropped` 不再有。状态行按上面的格式。
- 单测：mtime 排序与整体字节、最后一条记录带走目录、跨模型"谁最旧谁走"（含同龄与自己清空）、一次取走的统计、状态行两种形状。
- 真机（独立 `LMK_HOME`，`max_size: 3G`，造一个"别的模型"目录 60×48 MiB、mtime 为 2020 年，真模型写 cache；M3 Ultra，社区 4-bit 27B，
  两条不同的 4,575-token 请求）：别的模型 2784 → 1632 MiB（被淘汰的恰是最旧的 k00–k25，剩下的最旧是 k26），本模型 0 → 1425 MiB，
  目录合计始终 ≤ 3072 MiB。每次存盘一条 Trimmed 日志（这次 15 条），偏碎，没合并。
- 没测：上限被磁盘空闲量压低的那条路径与跨模型同时发生；上线服务（200G）上的实际表现；残缺链恢复（另一件事）。
