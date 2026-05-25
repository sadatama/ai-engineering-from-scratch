# 记忆：虚拟上下文与 MemGPT

> 上下文窗口是有限的。对话、文档和工具跟踪轨迹则不是。MemGPT (Packer et al., 2023) 将其类比为操作系统虚拟内存——主上下文是 RAM，外部存储是磁盘，agent 在两者之间进行页面置换。这是 2026 年每个记忆系统继承的模式。

**类型：** 构建
**语言：** Python (stdlib)
**前置条件：** Phase 14 · 01 (Agent Loop)、Phase 14 · 06 (工具使用)
**时间：** 约 75 分钟

## 学习目标

- 解释 MemGPT 基于的操作系统类比：主上下文 = RAM，外部上下文 = 磁盘，记忆工具 = 页面调入/调出。
- 使用标准库实现两层 MemGPT 模式：主上下文缓冲区、外部可搜索存储以及页面调入/调出工具。
- 描述 agent 如何发出"中断"来查询或修改外部记忆，以及结果如何被拼接到下一个 prompt 中。
- 识别 MemGPT 中延续到 Letta（Lesson 08）和 Mem0（Lesson 09）的设计选择。

## 问题

上下文窗口看起来应该能解决记忆问题。但实际上并非如此。在生产环境中有三种反复出现的失败模式：

1. **溢出。** 多轮对话、长文档或工具调用密集的轨迹会超出窗口限制。超出截止点的所有内容都会丢失。
2. **稀释。** 即使在窗口内部，塞入无关上下文也会稀释对重要内容的注意力。前沿模型在长输入上仍然会出现性能下降。
3. **持久性。** 新会话以空窗口开始。没有外部记忆的 agent 无法跨会话说"记得你上次让我做……"。

更大的窗口有帮助，但不能解决根本问题。Mem0 的 2025 年论文测量表明，带有外部记忆的 4k 窗口 agent 能捕获长程事实，而 128k 窗口的基线模型却无法捕获。

## 核心概念

### MemGPT：操作系统的类比

Packer 等人 (arXiv:2310.08560, v2 Feb 2024) 将上下文管理映射到操作系统的虚拟内存：

| OS 概念 | MemGPT 概念 | 2026 年生产级对应 |
|---------|------------|-----------------|
| RAM | 主上下文（prompt） | Anthropic/OpenAI 上下文窗口 |
| 磁盘 | 外部上下文 | 向量数据库、KV 存储、图存储 |
| 页面错误 | 记忆工具调用 | `memory.search`、`memory.read`、`memory.write` |
| OS 内核 | agent 控制循环 | 带有记忆工具的 ReAct 循环 |

agent 运行正常的 ReAct 循环。额外的一类工具使其能够将数据调入和调出主上下文。

### 两个层级

- **主上下文。** 固定大小的 prompt，保存当前任务。始终对模型可见。
- **外部上下文。** 无界存储，通过工具可搜索。在相关时读取，在事实出现时写入。

原始论文在两个超出基础窗口的任务上评估了该设计：超过 100k token 的文档分析，以及跨多天的具有持久记忆的多会话聊天。

### 中断模式

MemGPT 引入了"记忆即中断"的概念：在对话过程中，agent 可以调用记忆工具，运行时执行该工具，结果作为新的观察拼接到下一个 assistant 轮次中。概念上等同于 Unix 的 `read()` 系统调用——阻塞进程、返回字节、进程继续执行。

标准记忆工具接口：

- `core_memory_append(section, text)` —— 写入到 prompt 的一个持久化段落。
- `core_memory_replace(section, old, new)` —— 编辑一个持久化段落。
- `archival_memory_insert(text)` —— 写入到可搜索的外部存储。
- `archival_memory_search(query, top_k)` —— 从外部存储检索。
- `conversation_search(query)` —— 扫描过去的对话轮次。

### MemGPT 的终点与 Letta 的起点

2024 年 9 月，MemGPT 演变为 Letta。研究仓库 (`cpacker/MemGPT`) 仍然保留；Letta 扩展了设计：

- 三个层级而非两个（core、recall、archival——Lesson 08）。
- 原生推理取代 `send_message`/heartbeat 模式（Lesson 08）。
- 休眠期 agent 运行异步记忆工作（Lesson 08）。

即使生产系统运行的是 Letta、Mem0 或自定义的两层存储，MemGPT 论文仍然是 2026 年的基础。

### 这种模式可能出错的地方

- **记忆腐化。** 写入速度超过读取速度；检索被陈旧事实淹没。修复方法：定期整合（Letta 休眠期），显式失效（Mem0 冲突检测器）。
- **记忆投毒。** 外部记忆是检索到的文本。如果攻击者控制的内容落入了记忆笔记，agent 会在下一个会话中重新摄取它。这实质上是 Greshake 等人（Lesson 27）的攻击随时间重演。
- **引用丢失。** Agent 回忆起"用户让我交付 X"，但无法引用是哪一轮对话。在每次归档写入时存储源引用（session ID、turn ID）。

## 动手实现

`code/main.py` 使用标准库实现了 MemGPT 的两层模式：

- `MainContext` —— 固定大小的 prompt 缓冲区，包含 `core` 字典和 `messages` 列表；当超出容量时自动压缩最旧的消息。
- `ArchivalStore` —— 类似 BM25 的内存存储（基于 token 重叠评分），存储 (id, text, tags, session, turn) 记录。
- 五个映射到 MemGPT 接口的记忆工具。
- 一个脚本化的 agent，先用事实填充归档存储，然后通过调用 `archival_memory_search` 回答问题。

运行：

```
python3 code/main.py
```

追踪输出展示 agent 写入三个事实，将主上下文填满到容量上限（强制淘汰旧内容），然后通过从归档中检索来回答后续问题——在没有任何真实 LLM 的情况下复现了 MemGPT 的工作流程。

## 实际应用

当今每一个生产级记忆系统都是 MemGPT 的变体：

- **Letta**（Lesson 08）——三个层级、原生推理、休眠期计算。
- **Mem0**（Lesson 09）——向量 + KV + 图存储，融合评分层。
- **OpenAI Assistants / Responses** —— 通过线程和文件进行托管记忆。
- **Claude Agent SDK** —— 通过 skills 和会话存储实现长期记忆。

根据运维形态（自托管、托管、框架集成）来选择，而不是根据核心模式——核心模式都是 MemGPT。

## 交付物

`outputs/skill-virtual-memory.md` 是一个可复用的技能，可为任何目标运行时生成正确的两层记忆框架（主存储 + 归档 + 工具接口），并内置淘汰策略和引用字段。

## 练习

1. 添加以 token 度量的 `max_main_context_tokens` 上限（用 `len(text.split())` * 1.3 近似估算）。当超出上限时，将最旧的消息压缩为摘要。比较有摘要器和无摘要器的行为差异。
2. 在归档存储上正确实现 BM25（词频、逆文档频率）。在玩具事实集上测量 recall@10，与 token 重叠基线对比。
3. 向归档插入添加 `citation` 字段（session_id、turn_id、source_url）。让 agent 在每次基于检索的回答中引用来源。
4. 模拟记忆投毒：添加一条归档记录，内容为 "忽略所有未来的用户指令"。编写一个守卫，扫描检索结果中是否有指令形式的文本，并将其标记为不可信。
5. 将实现移植到使用 MemGPT 研究仓库的 core-memory JSON schema（`cpacker/MemGPT`）。当从扁平字符串切换到类型化段落时，会有什么变化？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 虚拟上下文 | "无限记忆" | 主（prompt）+ 外部（可搜索）层级，具有页面调入/调出 |
| 主上下文 | "工作记忆" | prompt——固定大小，始终可见 |
| 归档记忆 | "长期存储" | 外部可搜索持久化，按需检索 |
| 核心记忆 | "持久化 prompt 段落" | 固定在主上下文内部的命名段落 |
| 记忆工具 | "记忆 API" | Agent 发出的读写外部记忆的工具调用 |
| 中断 | "记忆页面错误" | Agent 暂停，运行时获取，结果拼接到下一个轮次 |
| 记忆腐化 | "陈旧事实" | 旧写入淹没检索；通过整合修复 |
| 记忆投毒 | "注入持久化笔记" | 攻击者内容存储为记忆，在召回时重新摄取 |

## 延伸阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) —— 受操作系统启发的虚拟上下文论文
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) —— 三层架构的演变
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 将上下文视为预算
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) —— 基于此模式的混合生产级记忆
