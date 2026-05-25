# 工具使用与 Function Calling

> Toolformer (Schick et al., 2023) 开启了自监督工具标注。Berkeley Function Calling Leaderboard V4 (Patil et al., 2025) 设定了 2026 年的评估标准：40% agentic、30% 多轮、10% 实时、10% 非实时、10% 幻觉检测。单轮已基本解决。记忆、动态决策和长程工具链条尚未解决。

**类型：** 构建
**语言：** Python (stdlib)
**前置条件：** Phase 14 · 01 (Agent Loop)、Phase 13 · 01 (Function Calling 深入)
**时间：** 约 60 分钟

## 学习目标

- 解释 Toolformer 的自监督训练信号：仅当工具执行结果降低了下一个 token 的损失时，才保留工具标注。
- 列举 BFCL V4 的五个评估类别及其各自衡量的内容。
- 使用标准库实现一个具备 schema 验证、参数强制转换和执行沙箱的工具注册表。
- 诊断 2026 年的三大开放问题：长程工具链、动态决策和记忆。

## 问题

早期的工具使用关注的是：模型能否预测出正确的函数调用？现代工具使用关注的是：模型能否在 40 步的链条中串联工具，具备记忆能力，处理部分可观测性，从工具失败中恢复，且不会虚构不存在的工具？

Toolformer 奠定了基线：模型可以通过自监督学习何时调用工具。BFCL V4 定义了 2026 年的评估目标。两者之间的差距正是生产级 agent 实际驻留的空间。

## 核心概念

### Toolformer (Schick et al., NeurIPS 2023)

核心思想：让模型在自己的预训练语料库中标注候选 API 调用。对每个候选调用，执行它。仅当包含工具结果能降低下一个 token 的损失时，才保留该标注。在过滤后的语料库上进行微调。

涉及的工具：计算器、问答系统、搜索引擎、翻译器、日历。自监督信号完全取决于工具是否有助于预测文本——不需要人工标注。

规模效应：工具使用能力随规模涌现。较小的模型会因为工具标注而性能下降；较大的模型则受益。这就是为什么 2026 年的前沿模型在工具使用方面表现强劲，而大多数 7B 模型需要显式的工具使用微调才能达到可靠水平。

### Berkeley Function Calling Leaderboard V4 (Patil et al., ICML 2025)

BFCL 是 2026 年事实上的评估标准。V4 的组成：

- **Agentic（40%）**——完整 agent 轨迹：记忆、多轮、动态决策。
- **多轮（30%）**——带有工具链的交互式对话。
- **实时（10%）**——用户提交的真实 prompt（更困难的分布）。
- **非实时（10%）**——合成测试用例。
- **幻觉检测（10%）**——检测何时不应调用任何工具。

V3 引入了基于状态的评估：在工具序列之后，检查 API 的实际状态（例如 "文件是否已创建？"），而不是匹配工具调用的 AST。V4 新增了网络搜索、记忆和格式敏感性等类别。

2026 年关键发现：单轮 function calling 已接近解决。失败主要集中在记忆（跨轮次携带上下文）、动态决策（基于先前结果选择工具）、长程链条（20+ 步后出现漂移）以及幻觉检测（在无工具适用时拒绝调用）。

### Tool Schema

每个提供商都有自己的 schema。它们在细节上有所不同，但共享相同的结构：

```
name: string
description: string（功能是什么，何时使用）
input_schema: JSON Schema（properties、required、types、enums）
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者都接受 JSON Schema。描述是关键信息——模型阅读描述来选择正确的工具。糟糕的工具描述是导致"选错工具"失败的首要原因。

### 参数验证

不要信任任何工具调用。需要验证：

1. **类型强制转换。** 模型可能返回字符串 "5"，而 schema 要求 int。如果不模糊则强制转换；如果模糊则拒绝。
2. **枚举验证。** 如果 schema 规定 `status in {"open", "closed"}`，而模型发出了 `"in_progress"`，则用描述性错误拒绝。
3. **必填字段。** 缺失必填字段 -> 立即将错误观察返回给模型，而不是崩溃。
4. **格式验证。** 日期、邮箱、URL——使用具体的解析器验证，而不是正则表达式。

每次验证失败都应返回结构化的观察结果，以便模型可以用正确的格式重试。

### 并行工具调用

现代提供商支持在一个 assistant 轮次中并行调用多个工具。循环如下：

1. 模型发出 3 个工具调用，具有不同的 `tool_use_id`。
2. 运行时执行这些调用（如果互不依赖则并行执行）。
3. 每个结果作为 `tool_result` 块传回，通过 `tool_use_id` 进行关联。

工程规则：将关联 ID 视为关键信息。如果交换它们，会导致"错误工具到错误结果"的路由。

### 沙箱

工具执行是沙箱边界。详见 Lesson 09。简要版本：每个工具都应指定读/写范围、网络访问、超时时间和内存上限。通用的 `run_shell(cmd)` 是危险信号；具体的 `git_status()` 更安全。

## 动手实现

`code/main.py` 实现了生产级别的工具注册表：

- JSON Schema 子集验证器（仅使用标准库）。
- 具备描述、输入 schema、超时和执行器的工具注册。
- 参数强制转换和枚举验证。
- 带有关联 ID 的并行工具调度。
- 作为结构化字符串的错误观察。

运行：

```
python3 code/main.py
```

追踪输出展示了一个 mini agent 在一个轮次中调用三个工具，其中一个故意构造的错误调用被拒绝，并返回模型可以处理的描述性错误。

## 实际应用

每个提供商都有自己的工具 schema——Anthropic、OpenAI、Gemini、Bedrock。如果需要多提供商支持，使用翻译层（OpenAI Agents SDK、Vercel AI SDK、LangChain tool adapter）。BFCL 是参考基准——如果工具使用是产品的核心功能，在发布前用它测试你的 agent。

## 交付物

`outputs/skill-tool-registry.md` 为给定的任务领域生成工具目录、schema 和注册表。包含描述质量检查（每个工具的描述是否告诉模型何时使用它？）。

## 练习

1. 添加一个"no-op"工具，让模型可以显式拒绝使用任何其他工具。在类似 BFCL 的幻觉检测测试上测量效果。
2. 实现 int-as-string 和 float-as-string 的参数强制转换。强制转换在什么情况下开始掩盖真正的 bug？
3. 添加每个工具的超时和熔断器（连续 3 次失败后，在 60 秒内拒绝该工具）。这对模型的恢复策略有何改变？
4. 阅读 BFCL V4 描述。选择一个类别（例如 "多轮"），通过你的 agent 运行 10 个示例 prompt。报告通过率。
5. 将标准库验证器移植到 Pydantic 或 Zod。Pydantic/Zod 捕获了什么而玩具版本没有捕获到？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Function calling | "工具使用" | 具备验证 schema 的结构化输出工具调用 |
| Toolformer | "自监督工具标注" | Schick 2023——仅保留其执行结果降低下一个 token 损失的工具调用 |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 年基准：40% agentic、30% 多轮、10% 实时、10% 非实时、10% 幻觉检测 |
| Tool schema | "模型的函数签名" | name、description、参数的 JSON Schema |
| tool_use_id | "关联 ID" | 将工具调用与其结果绑定；对并行调度至关重要 |
| 幻觉检测 | "知道何时不应调用" | V4 类别：当没有工具适用时拒绝调用 |
| 参数强制转换 | "字符串到整数的修复" | 针对可预测的 schema 不匹配进行窄范围修复；模糊时拒绝 |
| 沙箱 | "工具执行边界" | 每个工具的读/写范围、网络、超时、内存上限 |

## 延伸阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) —— 自监督工具标注
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) —— 2026 年评估基准
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) —— Claude Agent SDK 中的生产级工具 schema
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) —— function tool 类型和 Guardrails
