# Anthropic 工作流模式：简单优于复杂

> Schluntz 和 Zhang（Anthropic，2024年12月）将工作流（预定义路径）与 Agent（动态工具使用）区分开来。五种工作流模式覆盖了大多数场景。从直接 API 调用开始。只有在步骤无法预测时才引入 Agent。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 第十四阶段 · 01（Agent 循环）
**时间：** 约 60 分钟

## 学习目标

- 说出 Anthropic 的五种工作流模式：Prompt Chaining（提示链式调用）、Routing（路由）、Parallelization（并行化）、Orchestrator-Workers（编排器-工作者）、Evaluator-Optimizer（评估器-优化器）。
- 解释 Agent 与工作流的区别以及各自的工程成本。
- 识别何时选择工作流而非 Agent（以及相反的情况）。
- 使用标准库在脚本化 LLM 上实现全部五种模式。

## 问题

团队往往为只需单个函数调用即可解决的问题引入多 Agent 框架。这种成本是真实存在的：框架增加了额外的层次，模糊了提示词、隐藏了控制流，并引入了过早的复杂性。Schluntz 和 Zhang 在 2024 年 12 月的博文是业界引用最多的反驳之声：从简单开始，只有在复杂性带来足够收益时才引入它。

## 概念

### 工作流 vs Agent

- **工作流。** 通过预定义的代码路径编排 LLM 和工具。工程师掌控执行图。
- **Agent。** LLM 动态地指导自己的工具使用并自主决定步骤。模型掌控执行图。

两者各有其用。工作流更便宜、更快、更易于调试。Agent 能够解决开放式问题，但故障模式更难分析。

### 增强型 LLM

所有五种模式的基础：一个 LLM 配备三种能力——搜索（检索）、工具（操作）、记忆（持久化）。任何 API 调用都可以使用这些能力。

### 五种模式

1. **Prompt Chaining（提示链式调用）。** 调用 1 的输出作为调用 2 的输入。适用于任务可以被清晰地线性分解的场景。步骤之间可以有可选的程序化门控。

2. **Routing（路由）。** 一个分类器 LLM 决定调用哪个下游 LLM 或工具。适用于不同类别的输入需要不同处理方式的场景（一级技术支持 vs 退款 vs Bug 报告 vs 销售咨询）。

3. **Parallelization（并行化）。** 同时运行 N 个 LLM 调用，聚合结果。有两种形式：分段（处理不同数据块）和投票（相同提示词，N 次运行，取多数/综合结果）。

4. **Orchestrator-Workers（编排器-工作者）。** 一个编排器 LLM 动态决定运行哪些工作者（也是 LLM）并综合它们的输出。类似于 Agent 循环，但编排器不会无限循环。

5. **Evaluator-Optimizer（评估器-优化器）。** 一个 LLM 提出答案，另一个 LLM 评估它。迭代直到评估器通过。这是 Self-Refine（第 05 课）的泛化形式。

### 工作流优于 Agent 的场景

- **可预测的任务。** 如果你能枚举出步骤，就应该枚举。
- **成本受限的任务。** 工作流的步骤数是有限的；Agent 可能无限循环。
- **合规受限的任务。** 审计人员想要阅读执行图，而不是从轨迹中推断执行图。

### Agent 优于工作流的场景

- **开放式研究。** 下一步取决于上一步返回了什么。
- **可变长度的任务。** 从几分钟到几小时的工作，步骤数量未知。
- **新领域。** 当你还不知道正确的工作流时——先探索，后固化。

### 上下文工程配套学科

"面向 AI Agent 的有效上下文工程"（Anthropic 2025）系统化了相邻学科：200k 上下文窗口是一个预算，而不是一个容器。何时包含什么、何时压缩、何时让上下文增长。详见第十四阶段关于上下文压缩的课程（本课程体系中重新编号前的第十四阶段第 06 课）。

## 构建

`code/main.py` 在 `ScriptedLLM` 上实现了全部五种工作流模式：

- `prompt_chain(input, steps)` — 顺序执行。
- `route(input, classifier, handlers)` — 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` — N 次运行，聚合。
- `orchestrator_workers(task, workers)` — 编排器挑选工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — 循环直到通过。

运行：

```
python3 code/main.py
```

每个模式都会打印其执行轨迹。每个模式的代码行数约为 10-15 行；而一个框架的成本以千行为单位计算。

## 使用

- 大多数任务使用直接 API 调用。
- 只有当模式真正需要持久状态（LangGraph）、Actor 模型并发（AutoGen v0.4）或角色模板化（CrewAI）时，才引入框架。
- 当你想获得 Claude Code harness 形态而不需要从头重建时，使用 Claude Agent SDK。

## 交付

`outputs/skill-workflow-picker.md` 根据给定的任务描述选择正确的模式，包括决策理由以及在工作流不足时向 Agent 的重构路径。

## 练习

1. 实现带置信度阈值的路由。低于阈值 -> 升级到人工处理。对于一级技术支持场景，阈值应设在什么位置？
2. 为 `parallel_vote` 添加超时机制。当一个调用挂起时会发生什么？如何在有缺失投票的情况下进行聚合？
3. 将 `evaluator_optimizer` 改为 bandit 模式：在迭代过程中保留前 2 个最佳输出，这样后期的好结果不会被后期的坏结果覆盖。
4. 将 Prompt Chaining 与 Routing 结合：一个路由器选择三条链中的一条。测量 token 成本并与使用单个大提示词的方案对比。
5. 选择一个你的生产功能。画出工作流图。数一下步骤数量。在这里使用 Agent 是否真的更好？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|------------------------|
| Workflow（工作流） | "预定义流程" | 工程师拥有的 LLM 和工具调用图 |
| Agent（智能体） | "自主 AI" | 模型拥有的图；动态工具调度 |
| Augmented LLM（增强型 LLM） | "带工具的 LLM" | LLM + 搜索 + 工具 + 记忆；原子单元 |
| Prompt Chaining（提示链式调用） | "顺序调用" | 调用 N 的输出是调用 N+1 的输入 |
| Routing（路由） | "分类器分发" | 选择哪条链/哪个模型处理输入 |
| Parallelization（并行化） | "扇出" | N 个并发调用；通过分段或投票聚合 |
| Orchestrator-Workers（编排器-工作者） | "调度器 Agent" | 编排器 LLM 动态选择专业 LLM |
| Evaluator-Optimizer（评估器-优化器） | "提案者 + 裁判" | 迭代直到评估器通过；Self-Refine 的泛化形式 |

## 进一步阅读

- [Anthropic, Building Effective Agents (2024年12月)](https://www.anthropic.com/research/building-effective-agents) — 五种工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 配套学科
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 当有状态图值得其成本时
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — Orchestrator-Workers 模式的产品化实现
