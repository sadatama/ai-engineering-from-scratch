# 评估驱动的 Agent 开发

> Anthropic 的指导方针："从简单的 prompt 开始，通过全面的评估优化它们，只有在必要时才添加多步骤的 agentic 系统。"评估不是最后一步，而是驱动 Phase 14 中每一个其他选择的外层循环。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** Phase 14 全部内容。
**时间：** 约 60 分钟

## 学习目标

- 说出三个评估层次——静态 benchmark、自定义离线评估、在线生产评估——以及各自的作用。
- 解释 evaluator-optimizer（评估器-优化器）的紧密循环。
- 描述 2026 年最佳实践：评估代码与产品代码放在一起，在 CI 中运行，控制 PR 的合入。
- 将 Phase 14 的每一课与其生成的评估用例联系起来。

## 问题

Agent 在 demo 中表现良好，但在生产环境中以 demo 无法预测的方式失败。Benchmark 回答的是"这个模型整体能力如何？"而不是"这个 agent 是否为我的产品提交了正确的补丁？"答案在于：三个层次的评估，持续运行，每个 guardrail 和学到的规则都映射到一个评估用例。

## 概念

### 三个评估层次

1. **静态 benchmark** — 代码领域用 SWE-bench Verified（第 19 课），浏览器/桌面操作用 WebArena/OSWorld（第 20 课），通用能力用 GAIA（第 19 课），工具使用用 BFCL V4（第 06 课）。用于跨模型比较和回归拦截。数据污染是真实存在的：SWE-bench+ 发现了 32.67% 的答案泄漏。务必报告 Verified / +-audited 分数。

2. **自定义离线评估** — 针对你的产品形态：
   - LLM-as-judge（Langfuse、Phoenix、Opik — 第 24 课）。
   - 基于执行的评估（运行补丁，检查测试）。
   - 基于轨迹的评估（将动作序列与黄金标准对比；OSWorld-Human 显示顶级 agent 的步数是黄金标准的 1.4-2.7 倍）。

3. **在线评估** — 生产环境：
   - 会话回放（Langfuse）。
   - Guardrail 触发的告警（第 16、21 课）。
   - 每步成本/延迟追踪（第 23 课 OTel span）。

### Evaluator-Optimizer（评估器-优化器，Anthropic）

紧密循环：

1. 提案者生成输出。
2. 评估器进行评判。
3. 不断优化直到评估器通过。

这是 Self-Refine（第 05 课）的泛化形式。任何你关心的 agent 流程都可以用 evaluator-optimizer 包装以提高可靠性。

### 2026 年最佳实践

- 评估代码与产品代码放在一起。
- 在每个 PR 上通过 CI 运行。
- 根据评估分数控制合入（例如"相对 main 分支的回归不超过 5%"）。
- 每个 guardrail 映射到一个评估用例。
- 每个学到的规则（Reflexion、pro-workflow learn-rule）映射到一个失败用例。

### 串联 Phase 14

Phase 14 中的每一课都生成评估用例：

| 课程 | 生成的评估用例 |
|------|--------------|
| 01 Agent Loop | 预算耗尽、无限循环防护 |
| 02 ReWOO | 工具失败时 Planner 正确重新规划 |
| 03 Reflexion | 学到的反思在重试时被应用 |
| 05 Self-Refine/CRITIC | Judge 通过优化后的输出 |
| 06 工具使用 | 参数强制转换正确；未知工具被拒绝 |
| 07-10 记忆 | 检索引用匹配来源；过时的事实被判定无效 |
| 12 工作流模式 | 每种模式产生正确的输出 |
| 13 LangGraph | 恢复操作精确复现状态 |
| 14 AutoGen Actors | DLQ 捕获崩溃的处理器 |
| 16 OpenAI Agents SDK | Guardrail 在正确的输入上触发 |
| 17 Claude Agent SDK | Subagent 结果返回给 orchestrator |
| 19-20 Benchmark | SWE-bench Verified 分数、WebArena 成功率、OSWorld 效率 |
| 21 Computer Use | 每步安全检查捕获注入的 DOM |
| 23 OTel | Span 发出必需的属性 |
| 26 失败模式 | 检测器标记已知的失败 |
| 27 Prompt Injection | PVE 拒绝被污染的检索结果 |
| 28 编排 | Supervisor 路由到正确的专业 agent |
| 29 运行时形态 | DLQ 处理 N% 的失败 |

如果你的评估套件为每个条目都有对应的用例，你就覆盖了 Phase 14 的全部内容。

### 评估驱动开发可能失败的地方

- **没有基线。** 没有 last-known-good 的评估结果是不可读的。存储基线。
- **LLM-judge 没有 grounding。** Judge 也会产生幻觉。CRITIC 模式（第 05 课）——judge 依赖外部工具作为 grounding。
- **过拟合评估。** 针对评估的优化偏离了生产实用性。轮换用例。
- **不稳定的评估。** 非确定性用例导致误报。固定随机种子，快照状态。

## 动手构建

`code/main.py` 是一个标准库评估框架：

- 带分类的用例注册表（benchmark、自定义、在线）。
- 一个受测的脚本化 agent。
- Evaluator-optimizer 循环：提案、评判、优化，直到通过或达到最大轮次。
- CI 门禁：汇总通过率 + 相对基线的回归检测。

运行方式：

```
python3 code/main.py
```

输出：每个用例的通过/失败、回归标记、CI 门禁判定。

## 实际使用

- 在与 agent 代码同一个仓库中编写评估用例。
- 在每次 PR 中通过 CI 运行它们。
- 在有回归时让构建失败。
- 随时间追踪通过率。
- 每次生产故障都对应一个新的用例。

## 交付产出

`outputs/skill-eval-suite.md`：为一个 agent 产品构建三层评估套件，包含 CI 门禁和回归追踪。

## 练习

1. 选择你的一个生产故障，编写一个能复现它的评估用例。你的 agent 现在能通过吗？
2. 为你的领域构建一个 LLM-judge 评分标准，包含三个维度（事实性、语气、范围）。对 50 个会话打分。
3. 将评估套件接入 CI。在回归 >= 5% 时让构建失败。
4. 添加一个轨迹效率指标：agent 的步数与黄金轨迹的步数相比如何？
5. 将 Phase 14 的每一课映射到你的评估套件中的一个用例。有遗漏的吗？那就是需要弥补的差距。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 静态 benchmark | "现成的评估" | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| 自定义离线评估 | "领域评估" | LLM-as-judge / 执行 / 轨迹，针对你的产品形态 |
| 在线评估 | "生产环境评估" | 会话回放、guardrail 告警、成本/延迟追踪 |
| Evaluator-optimizer | "提案-评判-优化" | 不断迭代直到 judge 通过 |
| CI 门禁 | "合入拦截" | 评估出现回归时让构建失败 |
| 基线 | "last-known-good" | 用于检测回归的参考分数 |
| 轨迹效率 | "步数相对黄金标准" | Agent 步数除以人类专家的最少步数 |

## 扩展阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "从简单开始，用评估优化"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 精心整理的 benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — 工具使用 benchmark
- [Langfuse 文档](https://langfuse.com/) — 实践中的评估 + 会话回放
