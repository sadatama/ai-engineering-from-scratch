# 学习进度追踪

> 最后更新：2026-05-26
> 学习计划：[week1-agent-interview-sprint.md](week1-agent-interview-sprint.md)

---

## Sprint 总进度

| Day | 主题 | 计划日期 | 完成日期 | 测验得分 | 记录 |
|:---:|------|:---:|:---:|:---:|:---:|
| 1 | Agent 内核三件套 | 周一 | 05-25 / 05-26 | Loop 10/16, Tool 10/16, Memory 导读 8/8 | ✅ |
| 2 | RAG 全链路 | 周二 | — | — | — |
| 3 | Agent 推理 + 记忆深度 | 周三 | — | — | — |
| 4 | Safety Harness + Prompt Injection | 周四 | — | — | — |
| 5 | 编排模式 + 失败模式 + 评估 | 周五 | — | — | — |
| 6 | 协议层 + 框架地图 | 周六 | — | — | — |
| 7 | 项目深挖 + 模拟面试 | 周日 | — | — | — |

### Day 2 提前启动

| 课程 | 完成日期 | 模式 | 记录 |
|------|:---:|:---:|:---:|
| Embeddings & Vector Representations (Phase 11 · 04) | 05-26 | 导读自答 8/8 | ✅ |

---

## Day 1 完成详情

### 已完成课程

#### 1. The Agent Loop（Phase 14 · Lesson 01）
- 文档：`phases/14-agent-engineering/01-the-agent-loop/docs/`
  - en.md（已读）| zh.md（已翻译）| quiz-review.md（错题本）
- 代码：`phases/14-agent-engineering/01-the-agent-loop/code/main.py`（已读）
- 测验得分：**10/16**（及格）
- 强项：ReAct 三要素、异常处理设计、停止条件安全性、Native Reasoning 演进、Trust Boundary
- 弱项：
  1. 五大要素记忆不牢（message buffer / tool registry / stop condition / turn budget / observation formatter）
  2. `tool_use_id` 与并行调用乱序匹配
  3. ToyLLM → 真实 API 替换后 Transcript 变化
- 错题本：`phases/14-agent-engineering/01-the-agent-loop/docs/quiz-review.md`

#### 2. Tool Use and Function Calling（Phase 14 · Lesson 06）
- 文档：`phases/14-agent-engineering/06-tool-use-and-function-calling/docs/`
  - en.md（已读）| zh.md（已翻译）| quiz-review.md（错题本）
- 代码：`phases/14-agent-engineering/06-tool-use-and-function-calling/code/main.py`（已读）
- 测验得分：**10/16**（及格）
- 强项：BFCL V4 评估体系、Tool Schema 设计、参数校验流程、并行调度理解、Hallucination 检测设计
- 弱项：
  1. Toolformer 自监督训练机制（过滤条件 + scale effect 归因错误）
  2. Sandbox 攻击面本质（混淆了层级隔离和参数自由度）
  3. 推理基础设施 × Agent 交叉视角未发挥专业深度
- 错题本：`phases/14-agent-engineering/06-tool-use-and-function-calling/docs/quiz-review.md`

#### 3. Memory: Virtual Context and MemGPT（Phase 14 · Lesson 07）
- 日期：2026-05-26
- 文档：`phases/14-agent-engineering/07-memory-virtual-context-memgpt/docs/`
  - en.md（已读）| zh.md（已翻译）| quiz-review.md（学习记录）
- 代码：`phases/14-agent-engineering/07-memory-virtual-context-memgpt/code/main.py`（已读）
- 模式：导读自答（8 题）
- 状态：8/8 通过
- 弱项：Page Fault 精确表述、Sleep-time agents 解决 memory rot
- 亮点：Memory poisoning ↔ trust boundary 概念拉通
- 学习记录：`phases/14-agent-engineering/07-memory-virtual-context-memgpt/docs/quiz-review.md`

### Day 1 完成状态
- [x] The Agent Loop
- [x] Tool Use and Function Calling
- [x] Memory: Virtual Context and MemGPT
- **Day 1 全部完成**

---

## Day 2 提前启动

### 已完成课程

#### 1. Embedding 与向量表示（Phase 11 · Lesson 04）
- 日期：2026-05-26
- 文档：`phases/11-llm-engineering/04-embeddings/docs/`
  - en.md（已读）| zh.md（已翻译·已读）| quiz-review.md（学习记录）
- 代码：`phases/11-llm-engineering/04-embeddings/code/embeddings.py`（已读）
- 模式：导读自答（8 题）
- 状态：8/8 通过
- 弱项：HNSW 精度损失精确机制、法律文档最优 chunking 策略
- 亮点：Matryoshka 截断→归一化的理解、Cascaded Retrieval 通用模式总结
- 学习记录：`phases/11-llm-engineering/04-embeddings/docs/quiz-review.md`

### Day 2 剩余课程
- [ ] RAG: Retrieval-Augmented Generation（Phase 11 · Lesson 06）
- [ ] Advanced RAG: Chunking, Reranking（Phase 11 · Lesson 07）

---

## 通用弱项模式

从两份错题本中提取的共性盲区：

1. **原理层深度不足**：能讲出"是什么"和"怎么用"，但被追问"为什么这样设计"时容易答偏
2. **协议/基础设施层细节缺失**：`tool_use_id`、sandbox attack surface、prompt caching 等生产级细节需要加强
3. **专业交叉视角未发挥**：三年推理部署经验是差异化优势，但在 Agent 相关问题中未主动调用

---

## 使用说明

将此文件与 [week1-agent-interview-sprint.md](week1-agent-interview-sprint.md) 放在同一目录下。

当 AI 助手读取本文件后：
1. 查看"Day X 剩余课程"了解未完成内容
2. 查看对应课程的 quiz-review.md 了解弱项
3. 按 Sprint 计划继续推进下一课

**下一步**：完成 Day 1 剩余课程 → Day 2 RAG 全链路
