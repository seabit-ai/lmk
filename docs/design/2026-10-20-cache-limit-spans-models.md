# `cache.max_size` 是全局上限：淘汰跨模型，状态行合并报告

日期 2026-10-20。状态：**裁决部分已下，未开工**（等 owner 的 "go"）。

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
