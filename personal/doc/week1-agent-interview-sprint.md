# 一周 Agent 面试冲刺 Sprint

> **目标**：一周内达到对 Agent 架构、RAG、Safety Harness 等热门面试问题的深度掌握
> **总课时**：~23h（5工作日×3h + 周日×8h）
> **学习方式**：每天精读指定课程文档 → 手写核心代码 → 整合进项目故事 → 自检面试问答
> **前置条件**：三年 AI 工程经验，熟悉 LLM 推理部署，ML/数学有遗忘但概念层面够用

---

## 课程路径索引

所有路径相对于仓库根目录 `phases/`。

### Day 1（周一）· Agent 内核三件套 — ~3h

> 核心认知：Agent = Loop + Tool + Memory。面试官问任何 Agent 问题，最终都落回这三个词。

| 时间 | 课程 | 路径 | 动作 |
|:---:|------|------|------|
| 40min | The Agent Loop | [01-the-agent-loop](phases/14-agent-engineering/01-the-agent-loop/) | 精读 docs/en.md，手写 agent_loop.py |
| 40min | Tool Use & Function Calling | [06-tool-use-and-function-calling](phases/14-agent-engineering/06-tool-use-and-function-calling/) | 精读，关注 tool schema 设计 |
| 40min | Memory: Virtual Context & MemGPT | [07-memory-virtual-context-memgpt](phases/14-agent-engineering/07-memory-virtual-context-memgpt/) | 精读，对比短期/长期/工作记忆 |
| 60min | 整合输出 | — | 在你的 demo 项目中画出 Agent Loop + Tool + Memory 架构图 |

**Day 1 面试自检**：
- "Agent Loop 怎么处理 tool call 失败？" → retry / fallback / human-in-the-loop 三层
- "Agent 上下文窗口不够了怎么办？" → sliding window / summarization / MemGPT virtual context

---

### Day 2（周二）· RAG 全链路 — ~3h

> 核心认知：RAG 是 Agent 的眼睛。从 chunking 到 reranking，每一步都要能讲出 trade-off。

| 时间 | 课程 | 路径 | 动作 |
|:---:|------|------|------|
| 40min | Embeddings & Vector Representations | [04-embeddings](phases/11-llm-engineering/04-embeddings/) | 精读，关注 embedding 选型 |
| 50min | RAG: Retrieval-Augmented Generation | [06-rag](phases/11-llm-engineering/06-rag/) | 精读 + 手写最简 RAG pipeline |
| 50min | Advanced RAG: Chunking, Reranking | [07-advanced-rag](phases/11-llm-engineering/07-advanced-rag/) | 精读 chunking + reranking + hybrid search |
| 40min | 整合输出 | — | 在 demo 项目中加入 RAG 模块设计 |

**Day 2 面试自检**：
- "Chunk size 怎么选？" → semantic chunking vs fixed-size，trade-off：recall vs precision
- "什么时候用 RAG 什么时候用 fine-tuning？" → RAG = 知识检索，FT = 行为塑形
- "RAG 的 failure mode 有哪些？" → 检索不相关 / 上下文污染 / 信息过时 / hallucination 放大

---

### Day 3（周三）· Agent 推理 + 记忆深度 — ~3h

> 核心认知：复杂任务需要 Planning + Self-Reflection。面试官问"Agent 怎么处理复杂多步任务"时，要能答出多种模式的区别。

| 时间 | 课程 | 路径 | 动作 |
|:---:|------|------|------|
| 45min | ReWOO & Plan-and-Execute | [02-rewoo-plan-and-execute](phases/14-agent-engineering/02-rewoo-plan-and-execute/) | 精读，对比 ReAct vs ReWOO |
| 45min | Reflexion & Verbal RL | [03-reflexion-verbal-rl](phases/14-agent-engineering/03-reflexion-verbal-rl/) | 精读，关注 self-reflection loop |
| 45min | Planning: HTN & Evolutionary | [11-planning-htn-and-evolutionary](phases/14-agent-engineering/11-planning-htn-and-evolutionary/) | 速读，了解至少 2 种 planning 方法 |
| 45min | 整合输出 | — | 给 demo 项目加入 Planning 模块设计 |

**Day 3 面试自检**：
- "ReAct 和 ReWOO 各适合什么场景？" → ReAct 适合不确定性高的任务，ReWOO 适合步骤明确的任务
- "Reflexion 怎么防止 agent 陷入自我否定循环？" → 需要 external feedback signal，不能纯自省

---

### Day 4（周四）· Safety Harness + Prompt Injection — ~3h

> 核心认知：安全是 Agent 面试的拉开分差领域。Prompt injection / indirect injection / jailbreak / many-shot 攻击，每种都要知道原理和防御。

| 时间 | 课程 | 路径 | 动作 |
|:---:|------|------|------|
| 50min | Prompt Injection & PVE Defense | [27-prompt-injection-defense](phases/14-agent-engineering/27-prompt-injection-defense/) | 精读，关注三层防御架构 |
| 40min | Indirect Prompt Injection | [15-indirect-prompt-injection](phases/18-ethics-safety-alignment/15-indirect-prompt-injection/) | 精读，理解攻击向量 |
| 40min | Red-Teaming: PAIR & Automated Attacks | [12-red-teaming-pair-automated-attacks](phases/18-ethics-safety-alignment/12-red-teaming-pair-automated-attacks/) | 速读，了解至少 3 种 red-team 策略 |
| 50min | 整合输出 | — | 设计 Safety Harness 三层架构 |

**Day 4 面试自检**：
- "你的 Agent 怎么防止 prompt injection？" → 输入隔离 + 最小权限 + 输出校验
- "Indirect vs Direct prompt injection？" → indirect 来自外部数据源（网页/文档），更难防御
- "什么是 PVE defense？" → Privilege-separated Virtual Environment

---

### Day 5（周五）· 编排模式 + 失败模式 + 评估 — ~3h

> 核心认知：这是面试官判断你"是不是真的在生产环境跑过 Agent"的关键。能讲 Agent Loop 的人很多，能讲清楚"Agent 在什么情况下会死、怎么救"的人很少。

| 时间 | 课程 | 路径 | 动作 |
|:---:|------|------|------|
| 45min | Orchestration Patterns | [28-orchestration-patterns](phases/14-agent-engineering/28-orchestration-patterns/) | 精读 Supervisor/Swarm/Hierarchical 三种模式 |
| 50min | Failure Modes: Why Agents Break | [26-failure-modes-agentic](phases/14-agent-engineering/26-failure-modes-agentic/) | **深度精读**——本周最有面试价值的单课 |
| 45min | Eval-Driven Agent Development | [30-eval-driven-agent-development](phases/14-agent-engineering/30-eval-driven-agent-development/) | 精读 Agent 评估三层体系 |
| 40min | 整合输出 | — | Failure Mode Checklist + Eval 方案 |

**Day 5 面试自检**：
- "Agent 最常见的 5 种失败模式？" → tool call loop / context overflow / hallucinated tool args / goal drift / premature termination
- "怎么评估一个 Agent 好不好？" → 分层：tool selection accuracy / plan quality / end-to-end success rate / cost per task

---

### Day 6（周六，部分时间）· 协议层 + 框架地图 — ~3h

> 核心认知：MCP 和 A2A 是 Agent 通信的基础协议。不需要写代码，但要能画架构图、讲清楚设计思路。

| 时间 | 课程 | 路径 | 动作 |
|:---:|------|------|------|
| 40min | MCP Fundamentals | [06-mcp-fundamentals](phases/13-tools-and-protocols/06-mcp-fundamentals/) | 速读，理解 MCP 设计哲学 |
| 30min | Building an MCP Server | [07-building-an-mcp-server](phases/13-tools-and-protocols/07-building-an-mcp-server/) | 浏览代码 |
| 30min | A2A Protocol | [19-a2a-protocol](phases/13-tools-and-protocols/19-a2a-protocol/) | 速读，对比 MCP vs A2A |
| 40min | Anthropic Workflow Patterns | [12-anthropic-workflow-patterns](phases/14-agent-engineering/12-anthropic-workflow-patterns/) | 精读 5 种 workflow pattern |
| 40min | 整合输出 | — | 画出项目完整技术架构图（A4 纸能放下） |

**Day 6 面试自检**：
- "MCP 和 A2A 的区别？" → MCP：人→工具（Client-Server），A2A：Agent→Agent（P2P）
- "5 种 Anthropic Workflow Pattern？" → Chain / Parallel / Router / Orchestrator-Worker / Evaluator-Optimizer

---

### Day 7（周日）· 项目深挖 + 模拟面试 — ~8h

> 核心认知：面试官不会考知识点，他会说"讲讲你做过的最有挑战的项目"——你要用一个故事把所有知识点串起来。

#### 上午（3h）：项目故事八维填充

对着以下 8 个维度，将本周所学"灌"进你的 demo 项目：

1. **业务背景** — 为什么要做这个 Agent？解决什么痛点？
2. **架构选型** — 为什么选这个 Loop 模式？trade-off 是什么？
3. **工具设计** — Agent 接了哪些 tool？tool schema 怎么设计的？
4. **记忆系统** — 短期/长期/工作记忆分别怎么实现？
5. **RAG 模块** — Chunking 策略？Embedding 选型？Reranking 方案？
6. **安全体系** — Input guard / Tool permission / Output verification 三层分别做了什么？
7. **评估体系** — 怎么衡量 Agent 质量？有哪些 metric？
8. **失败处理** — 遇到过什么 failure？怎么发现/怎么修的？

每一项至少能讲 2-3 分钟，有具体的技术决策 + trade-off 分析。

#### 下午（3h）：模拟面试十大高频问

**出声回答**（不要默念，说出来才算）：

1. 讲讲你做的这个 Agent 项目的整体架构 → 考察系统性思维
2. Agent Loop 里你是怎么处理错误的？ → 考察 Failure handling
3. RAG 的 chunk size 你选的多少？为什么？ → 考察工程细节
4. Agent 的安全你是怎么考虑的？ → 考察 Safety awareness
5. 你怎么评估这个 Agent 好不好用？ → 考察 Eval 体系
6. 如果重新做这个项目，你会改什么？ → 考察反思能力
7. Multi-agent 和 single-agent 的边界在哪？ → 考察架构判断力
8. MCP 和 A2A 分别解决什么问题？ → 考察协议理解
9. Agent 在生产环境最大的风险是什么？ → 考察生产经验
10. 最近关注哪些 Agent 方向的论文/项目？ → 考察技术视野

#### 晚上（2h）：查漏补缺

根据下午模拟面试中卡壳的地方，回扫对应课程：
- Multi-Agent 答不好 → [25-multi-agent-debate](phases/14-agent-engineering/25-multi-agent-debate/)
- 安全细节不够 → [29-moderation-systems](phases/18-ethics-safety-alignment/29-moderation-systems-openai-perspective-llamaguard/)
- 记忆系统讲不深 → [09-hybrid-memory-mem0](phases/14-agent-engineering/09-hybrid-memory-mem0/)

---

## 不学清单（明确砍掉）

| 砍掉的内容 | 理由 |
|------|------|
| Phase 1 数学全部 | 面试不会让你手推梯度，概念层面已够用 |
| Phase 2 ML 全部 | Agent 面试不考 ML 细节 |
| Phase 3 深度学习 | 日常工作已在用 |
| Phase 7 Transformers | 推理部署是老本行 |
| Phase 10 LLMs from Scratch | Agent 岗不考这个深度 |
| Phase 17 基础设施 | 老本行，不需要复习 |

---

## 产出清单（Sprint 验收标准）

- [ ] 1 个完整项目故事（八维填充完毕）
- [ ] 1 张 Agent 架构总图（面试可展示）
- [ ] 1 份 Failure Mode Checklist（8+ 种失败模式及对策）
- [ ] 10 道面试高频题的出声回答（录下来自己听一遍）
- [ ] 1 份 RAG 全链路架构设计
- [ ] 1 份 Safety Harness 三层防御设计

---

## Agent Failure Mode Checklist（核心参考）

以下是 Agent 系统最常见的 8+ 种失败模式，面试时至少要能讲出 5 种：

| # | 失败模式 | 现象 | 根因 | 缓解策略 |
|:---:|------|------|------|------|
| 1 | Tool Call Loop | Agent 无限循环调用同一个 tool | 缺少退出条件 / tool 返回格式不符合预期 | max_steps 限制 + loop detection + 连续相同 call 检测 |
| 2 | Context Overflow | 对话历史超出上下文窗口 | 长对话未做截断 | sliding window / summarization / MemGPT virtual context |
| 3 | Hallucinated Tool Args | Agent 传入了不存在的参数 | LLM 对 tool schema 理解偏差 | strict schema validation + retry with error feedback |
| 4 | Goal Drift | Agent 逐渐偏离原始任务目标 | 长链路中丢失原始意图 | 周期性 goal check + plan verification |
| 5 | Premature Termination | Agent 过早宣布任务完成 | 缺少完成条件验证 | 显式 verification step + 输出格式约束 |
| 6 | Tool Output Misinterpretation | Agent 错误理解 tool 返回结果 | tool 输出格式不一致 / 过于复杂 | 结构化 tool 输出 + 关键信息前置 |
| 7 | Prompt Injection | 外部内容劫持 Agent 行为 | 未隔离用户数据和系统指令 | 输入隔离 + 最小权限 + 输出校验 |
| 8 | Cascading Errors | 一步错误导致后续全错 | 错误未在早期被检测和纠正 | 每步验证 + 错误回滚 + 人类介入 |

---

## 快速参考：Agent 架构决策树

```
任务复杂度？
├── 简单单步任务 → ReAct Loop
│   └── Tool call + 直接返回
├── 中等多步任务 → ReWOO / Plan-Execute
│   └── 先规划全部步骤，再逐步执行
└── 复杂推理任务 → Tree of Thoughts / LATS
    └── 多路径搜索 + 剪枝 + 回溯

Agent 数量？
├── 单一 Agent → Supervisor 监控
├── 多 Agent 协作 → Orchestrator-Worker
└── 多 Agent 竞争 → Multi-Agent Debate / Swarm

记忆需求？
├── 单次对话 → 直接使用 conversation history
├── 跨会话记忆 → External vector store + RAG
└── 持久化知识 → Fine-tuning / skill library
```

---

## 学习进度追踪

在每个 Day 完成后打勾，记录完成日期和耗时：

| Day | 主题 | 完成 | 日期 | 实际耗时 | 备注 |
|:---:|------|:---:|------|:---:|------|
| 1 | Agent 内核三件套 | ☐ | ___ | ___ | |
| 2 | RAG 全链路 | ☐ | ___ | ___ | |
| 3 | Agent 推理 + 记忆深度 | ☐ | ___ | ___ | |
| 4 | Safety Harness + Prompt Injection | ☐ | ___ | ___ | |
| 5 | 编排模式 + 失败模式 + 评估 | ☐ | ___ | ___ | |
| 6 | 协议层 + 框架地图 | ☐ | ___ | ___ | |
| 7 | 项目深挖 + 模拟面试 | ☐ | ___ | ___ | |

---

> **使用说明**：将此文件拉取到公司电脑后，每天按 Day 顺序推进。完成一天的内容后，在进度追踪表中打勾。遇到问题时，告诉 AI 助手你当前的 Day 和进度，即可无缝继续学习。
>
> 相关课程文档的中文翻译版见各课程目录下的 `docs/zh.md`。
>
> 创建日期：2026-05-25
