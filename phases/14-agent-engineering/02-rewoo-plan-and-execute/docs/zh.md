# ReWOO 与 Plan-and-Execute：解耦式规划

> ReAct 将思考与行动交错在单一流中。ReWOO 将它们分离：先做一个大规划，然后执行。Token 用量减少 5 倍，HotpotQA 准确率提升 4%，而且你可以将 planner 蒸馏到 7B 模型。Plan-and-Execute 将其泛化；Plan-and-Act 将其扩展到网页导航。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** Phase 14 · 01（Agent Loop）
**时间：** 约 60 分钟

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 分离比 ReAct 的交错循环节省 token 并提高鲁棒性。
- 实现一个规划 DAG、一个按依赖排序的执行器和一个组合 worker 输出的 solver——全部使用标准库。
- 使用 2026 年"五种工作流模式"框架（Anthropic），判断一个任务应该以 plan-then-execute 方式还是交错 ReAct 方式运行。
- 识别何时需要 Plan-and-Act 的合成规划数据来处理长周期的网页或移动端任务。

## 问题

ReAct 交错式的 thought-action-observation 循环简单且灵活，但每次工具调用都必须携带完整的先前上下文——包括每一个之前的思考。Token 使用量随深度呈二次增长。更糟的是：当工具在循环中途失败时，模型必须从错误观察中重新推导整个计划。

ReWOO（Xu 等人，arXiv:2305.18323，2023 年 5 月）注意到了这一点并做出了一种选择：先一次性规划全部内容，并行获取证据，最后组合答案。一次 LLM 调用进行规划，N 次工具调用获取证据（可以并行），一次 LLM 调用进行求解。代价是更少的灵活性（计划是静态的），但换来了更好的 token 效率和更清晰的失败模式。

## 概念

### 三种角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        （工具调用，可能并行）
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 生成一个 DAG。每个节点指定一个工具、其参数以及它依赖的哪些前置节点（引用如 `#E1`、`#E2`）。Workers 按拓扑顺序执行节点。Solver 将所有内容整合在一起。

### 为什么 token 减少 5 倍

ReAct 的 prompt 长度随步数线性增长。在第 10 步时，prompt 包含 thought 1 加 action 1 加 observation 1 加 thought 2 加 action 2 加 observation 2，以此类推。每个中间步骤还冗余包含原始 prompt。

ReWOO 支付一次 planner prompt（较大）、N 次小型 worker prompt（每次只是工具调用，没有链条）和一次 solver prompt。在 HotpotQA 上，论文测量到约 5 倍的 token 减少，同时绝对准确率提升 4 个百分点。

### 为什么它更鲁棒

如果 worker 3 在 ReAct 中失败，循环必须在过程中从错误中推理出来。在 ReWOO 中，worker 3 返回一个错误字符串；solver 在原始计划的上下文中看到它，可以优雅降级。失败定位是按节点的，而不是按步的。

### Planner 蒸馏

论文的第二个结果：因为 planner 看不到观察结果，你可以用 175B 教师模型的 planner 输出来微调一个 7B 模型。小模型处理规划；大模型在推理时不需要。这现在已成为标准——许多 2026 年的生产 agent 使用小型 planner 和大型 executor，或反之。

### Plan-and-Execute（LangChain，2023）

LangChain 团队 2023 年 8 月的文章将 ReWOO 泛化为一个模式名称：Plan-and-Execute。前置 planner 输出步骤列表，executor 运行每一步，可选的 replanner 可以在观察结果后修订计划。这比 ReWOO 更接近 ReAct（replanner 将观察结果带回规划中），但保留了 token 节省。

### Plan-and-Act（Erdogan 等人，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将该模式扩展到长周期网页和移动 agent。关键贡献是合成规划数据：一个标注轨迹生成器产生规划显式可见的训练数据。用于微调 planner 模型，使其在 WebArena 类任务上持续运行超过 30-50 步，而单一的 ReAct 轨迹会失去连贯性。

### 何时选择哪种模式

| 模式 | 适用场景 |
|---------|------|
| ReAct | 短任务、环境未知、需要响应式异常处理 |
| ReWOO | 结构化任务且工具已知、对 token 敏感、可并行获取证据 |
| Plan-and-Execute | 类似 ReWOO 但在部分执行后需要重新规划 |
| Plan-and-Act | 长周期（>30 步）、网页/移动端/computer-use |
| Tree of Thoughts | 值得为搜索付出成本时（第 04 课） |

Anthropic 2024 年 12 月的指导：从最简单的开始。如果任务是一次工具调用加一个摘要，不要构建 ReWOO。如果任务是一个 40 步的研究任务，不要只用 ReAct。

## 构建它

`code/main.py` 实现了一个玩具级 ReWOO：

- `Planner`——一个脚本化策略，从 prompt 中输出规划 DAG。
- `Worker`——通过注册表调度每个节点的工具调用。
- `Solver`——脚本化组合，读取证据并生成最终答案。
- 依赖解析——像 `#E1` 这样的引用被替换为之前的 worker 输出。

演示回答了"What is the population of the capital of France, rounded to millions?"使用两步计划：（1）查找首都，（2）查找人口，然后求解。

运行它：

```
python3 code/main.py
```

轨迹首先显示完整计划，然后是 worker 结果，然后是 solver 组合。将 token 计数（我们打印一个粗略的字符计数）与 ReAct 风格的交错运行进行比较——在这种结构化任务上 ReWOO 胜出。

## 使用它

LangGraph 以配方形式提供了 Plan-and-Execute（`create_react_agent` 用于 ReAct，自定义图用于 plan-execute）。CrewAI 的 Flows 直接编码了该模式：你预先定义任务，Flow DAG 执行它们。Plan-and-Act 的合成数据方法主要还处于研究阶段；运行模式（显式规划 DAG）通过 LangGraph 和 CrewAI Flows 在生产中交付。

## 交付它

`outputs/skill-rewoo-planner.md` 在给定工具目录的情况下，从用户请求生成 ReWOO 规划 DAG。它在交给 executor 之前验证计划（无环、每个引用都已解析、每个工具都存在）。

## 练习

1. 为独立的规划节点并行化 worker 执行。在一个有 2 个并行组的 6 节点 DAG 上，这会带来什么好处？
2. 添加一个 replanner 节点，在任何 worker 返回错误时触发。将 ReWOO 变成 Plan-and-Execute 的最小改动是什么？
3. 将 `Planner` 替换为小模型（7B 级别），将 `Solver` 保留在 frontier 模型上。对比端到端质量——这种拆分在哪里会失败？
4. 阅读 ReWOO 论文第 4 节关于 planner 蒸馏的内容。概念性地复现 175B -> 7B 的结果：你需要什么训练数据，以及如何评估规划质量？
5. 将玩具实现移植到 Plan-and-Act 的轨迹形态：计划是一个序列，而不是 DAG。哪些权衡发生了变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| ReWOO | "无需观察的推理" | 先规划，然后并行获取证据，最后求解——planning prompt 中没有观察结果 |
| Plan-and-Execute | "LangChain 的 plan-execute 模式" | 在 ReWOO 基础上增加一个在执行后可选的 replanner 节点 |
| Plan-and-Act | "规模化 plan-execute" | 显式 planner/executor 分离，使用合成规划训练数据应对长周期任务 |
| Evidence reference | "#E1, #E2, ..." | 规划节点占位符，在调度时替换为前置 worker 输出 |
| Planner distillation | "小 planner，大 executor" | 用大型教师模型的 planner 轨迹微调小模型 |
| Token efficiency | "更少的往返" | 论文中在 HotpotQA 上相比 ReAct 减少 5 倍 token |
| DAG executor | "拓扑调度器" | 按依赖顺序运行规划节点；每一层级并行执行 |

## 延伸阅读

- [Xu 等人，ReWOO: Decoupling Reasoning from Observations（arXiv:2305.18323）](https://arxiv.org/abs/2305.18323)——规范论文
- [Erdogan 等人，Plan-and-Act（arXiv:2503.09572）](https://arxiv.org/abs/2503.09572)——使用合成规划的规模化 planner-executor
- [LangGraph Plan-and-Execute 教程](https://docs.langchain.com/oss/python/langgraph/overview)——框架配方
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——选择能有效工作的最简单模式
