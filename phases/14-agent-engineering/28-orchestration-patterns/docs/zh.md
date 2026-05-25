# 编排模式：Supervisor、Swarm、Hierarchical

> 四大编排模式在 2026 年的框架中反复出现：supervisor-worker（监督者-工作者）、swarm/peer-to-peer（群体/对等）、hierarchical（分层）、debate（辩论）。Anthropic 的指导方针是："关键在于为你的需求构建合适的系统。"从简单开始；只有当单个 agent 加上五种工作流模式还不够时，才引入拓扑结构。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** Phase 14 · 12（工作流模式），Phase 14 · 25（多 Agent 辩论）
**时间：** 约 60 分钟

## 学习目标

- 说出四种反复出现的编排模式及其各自适用的场景。
- 描述 2026 年 LangChain 建议：基于 tool-call 的监督 vs 使用 supervisor 库。
- 解释 Anthropic 的"构建合适的系统"原则，以及它如何决定拓扑选择。
- 使用标准库，针对一个通用的脚本化 LLM 实现全部四种模式。

## 问题

团队在需要"multi-agent"之前就急于采用它。四种模式在框架中反复出现；一旦你能说出它们的名字，就能选择合适的模式——或者完全跳过拓扑结构。

## 概念

### Supervisor-Worker（监督者-工作者）

- 一个中央路由 LLM 将任务分发给专业 agent。
- 决策：循环回自身、转交给专业 agent、终止。
- 专业 agent 之间不直接通信；所有路由都通过 supervisor 进行。

框架：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI Hierarchical Process。

**2026 年 LangChain 建议：** 通过直接的 tool call 实现监督，而不是使用 `create_supervisor`。这能提供更精细的上下文工程控制——你完全可以决定每个专业 agent 能看见什么。

### Swarm / Peer-to-Peer（群体 / 对等）

- Agent 通过共享的工具接口直接相互转交。
- 没有中央路由器。
- 比 supervisor 模式延迟更低（跳数更少）。
- 更难推理（没有单一控制点）。

框架：LangGraph swarm 拓扑、OpenAI Agents SDK handoffs（当所有 agent 可以互相转交时）。

### Hierarchical（分层）

- 监督者管理子监督者，子监督者管理工作者。
- 在 LangGraph 中实现为嵌套子图；在 CrewAI 中为嵌套 crew。
- 可以扩展到大规模 agent 群体，代价是运维复杂性增加。

适用场景：当单个 supervisor 的上下文预算无法容纳所有专业 agent 的描述时。

### Debate（辩论）

- 并行提案者 + 迭代交叉批评（第 25 课）。
- 与其说是编排，不如说是验证——但在框架中作为一种拓扑选择出现。

### CrewAI Crew vs Flow

CrewAI 将两种部署模式形式化：

- **Flow**：用于确定性事件驱动自动化（推荐作为生产环境的起点）。
- **Crew**：用于基于角色的自主协作。

这与上述四种模式是正交的，但可以映射到拓扑结构：Flow 通常是 supervisor 或 hierarchical；Crew 通常是带有 LLM 路由器的 supervisor。

### Anthropic 的指导方针

"在 LLM 领域的成功不在于构建最复杂的系统，而在于为你的需求构建合适的系统。"

决策顺序：

1. 单个 agent + 工作流模式（第 12 课）——从这里开始。
2. Supervisor-worker——当你有 2-4 个专业 agent 时。
3. Swarm——当延迟比推理清晰度更重要时。
4. Hierarchical——仅当 supervisor 上下文预算不足时。
5. Debate——当准确性比成本更重要时。

### 这种模式容易出错的地方

- **拓扑优先思维。** 在确定 multi-agent 要解决什么问题之前就认定"我们需要 multi-agent"。
- **Swarm 中的弹跳式转交。** A -> B -> A -> B。使用跳数计数器。
- **虚假的分层。** 三层架构因为"企业级"要求，但实际上只有两个团队。合并它们。

## 动手构建

`code/main.py` 使用标准库针对脚本化 LLM 实现了全部四种模式：

- `Supervisor` — 中央路由器。
- `Swarm` — 带直接转交的对等模式。
- `Hierarchical` — 监督者的监督者。
- `Debate` — 并行提案者 + 批评。

每种模式处理相同的三类意图任务（退款 / bug / 销售）。跟踪轨迹的形状不同。

运行方式：

```
python3 code/main.py
```

输出：每种模式的跟踪轨迹 + 操作计数。Supervisor 最清晰；Swarm 最短；Hierarchical 最深；Debate 最昂贵。

## 实际使用

- **LangGraph**：用于 supervisor 和 hierarchical（嵌套子图）。
- **OpenAI Agents SDK**：用于 handoff-as-tool（supervisor 形态）。
- **CrewAI Flow**：用于生产环境确定性流程。
- **自定义**：用于 debate 或需要精确控制的场景。

## 交付产出

`outputs/skill-orchestration-picker.md`：选择一种拓扑结构并实现它。

## 练习

1. 通过移除路由器，将 supervisor-worker 转换为 swarm。什么会出问题？什么会改善？
2. 为 swarm 添加跳数计数器：3 次转交后拒绝执行。它能捕获 A->B->A 的弹跳吗？
3. 为一个有 12 个专业 agent 的领域构建一个两级分层系统。如果不嵌套，上下文预算在哪里会失效？
4. 在接近生产的负载下对四种模式进行性能分析。每种模式在哪些指标上获胜（延迟、成本、准确性、可调试性）？
5. 阅读 Anthropic 的"Building Effective Agents"文章。将你的每个生产流程映射到四种模式之一。有没有无法清晰映射的？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Supervisor-worker | "路由器 + 专家" | 中央 LLM 将任务分发给专业 agent；它们之间不直接通信 |
| Swarm | "对等网络" | 通过共享工具直接转交；没有中央路由器 |
| Hierarchical | "监督者的监督者" | 为大规模群体嵌套子图 |
| Debate | "提案者 + 批评" | 并行提案者，交叉批评（第 25 课） |
| Tool-call-based supervision | "不用库的 Supervisor" | 将 supervisor 实现为直接 tool call，以获得上下文控制 |
| Crew | "自主团队" | CrewAI 的基于角色的协作模式 |
| Flow | "确定性工作流" | CrewAI 的事件驱动生产模式 |

## 扩展阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 五种模式 + agent 与工作流的区别
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor、swarm、hierarchical
- [CrewAI 文档](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate 模式
