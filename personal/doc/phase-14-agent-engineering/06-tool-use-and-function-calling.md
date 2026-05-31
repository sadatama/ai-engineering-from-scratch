# Tool Use and Function Calling · 错题本

> 测验日期：2026-05-26 | 得分：10/16 | 及格线：10/16
> 状态：压线及格，三个盲区需巩固

---

## 错题一：Toolformer 自监督训练机制（Q1 · 0.5/2）

**扣分原因**：将过滤条件理解为"梯度下降"，将 scale effect 归因为"推理速度"。两者均偏离了课程原文。

**正确内容**：

### Toolformer 的训练信号

Toolformer 的流程不是强化学习，而是**自监督数据清洗**：

```
1. 在预训练语料中插入候选 API 调用标记
2. 执行该 API 调用，获取工具返回结果
3. 用工具结果替换 API 调用标记
4. 计算替换前后的 next-token prediction loss
5. loss 下降 → 保留这条 tool annotation（工具确实帮助了语言建模）
6. loss 不降 → 丢弃（工具调用没用甚至有害）
```

**为什么不能保留所有标注？** 如果全保留，模型会学到"不管需不需要都调工具"——很多 API 调用返回的噪声反而干扰了语言建模。Toolformer 过滤掉了无用的工具调用，只留下真正有帮助的那些。

### Scale Effect 的正确解释

| 你的回答 | 正确解释 |
|------|------|
| "小模型推理快，tool call 拖慢速度" | 这是 latency 视角，不是 paper 的结论 |
| **正确原因** | **容量瓶颈（capacity bottleneck）**：小模型（<7B）参数不足以同时学好语言建模+工具调用，tool annotation 对它们是噪声。大模型有足够容量将工具调用内化为额外能力 |

Paper 原文：*"Smaller models hurt from tool annotations; larger models gain."* 这是模型容量的 trade-off，不是推理速度的 trade-off。

**面试话术**：

> "Toolformer 的核心创新是纯自监督的训练信号——不需要人类标注哪些场景该调工具。它用 next-token prediction loss 的变化来自动判断：工具调用是否真的帮助了语言建模。保留 loss 下降的，丢弃 loss 不降的。另外论文发现了一个重要的 scale effect：小模型容量不足以同时学好语言和工具，tool annotation 反而降低性能；大模型有足够容量将工具调用内化为能力。这也是为什么 2026 年 7B 以下的模型仍然需要专门的 tool-use fine-tuning。"

---

## 错题二：Sandbox 攻击面本质（Q7 · 1.0/2）

**扣分原因**：混淆了"层级隔离"和"攻击面（attack surface）"两个概念。

**正确内容**：

### `run_shell(cmd)` vs `git_status()` 的本质差异

| 维度 | `run_shell(cmd)` | `git_status()` |
|------|------|------|
| 参数 | 自由文本 `cmd`，任意命令 | 无用户可控参数 |
| 攻击面 | **极大**——注入攻击可执行任意 shell 命令 | **极小**——只做一件事 |
| 风险 | prompt injection 可直接控制服务器 | injection 无法改变 tool 行为 |

**本质**：不是"层级"问题，是**参数自由度**问题。一个接受自由文本参数的工具 = 一个攻击者可编程的执行入口。

### 三道防线（标准答案）

如果你必须提供 shell 执行能力：

1. **Schema 层**：限定可执行命令的白名单（`allowed_commands: ["git_status", "git_diff", "ls"]`），拒绝自由文本
2. **执行层**：Sandbox 限制——文件系统只读挂载、无网络访问、30s 超时、内存上限
3. **审计层**：所有执行的命令和参数写入不可篡改的审计日志，与 Agent trace 分离

**面试话术**：

> "`run_shell(cmd)` 的危险在于它把攻击面暴露给了自由文本参数——任何 prompt injection 都能通过 `cmd` 参数执行任意命令。正确的做法是把 shell 操作封装为具体的、无用户可控参数的工具函数，如 `git_status()`。如果确实需要 shell 能力，三道防线：白名单命令集 + sandbox 执行隔离 + 不可篡改审计日志。"

---

## 错题三：推理基础设施 × Agent 交叉视角（Q8 · 1.0/2）

**扣分原因**：作为三年推理工程师，只给出了"用 RAG 外置 memory"这种应用层建议，没有发挥出推理基础设施的专业深度。

**正确内容**：

### Memory 问题的推理瓶颈

Memory 问题的推理瓶颈**不是 max_tokens 不够用**，而是：

```
每轮 Agent 循环：
  ┌─ message buffer 膨胀（每轮追加 observation）
  ├─ prefill 时间线性增长：O(总 token 数)
  ├─ KV cache 命中率暴跌（后缀每轮都在变）
  └─ TTFT（Time To First Token）越来越慢
```

### 推理工程师应该给出的建议

**建议一：Prompt/Context Caching 固定 memory 前缀**

```
┌─────────────────────────────────────────────┐
│  Cached Prefix (不变)  │  New Turn (变化)     │
│  system prompt         │  latest observation  │
│  memory store          │  latest thought      │
│  tool definitions       │                       │
│  previous turns 1..N-1 │                       │
└─────────────────────────────────────────────┘
```

利用 Anthropic prompt caching 或 OpenAI automatic caching，将 memory + tool schema 固定在 cache 前缀。每轮只重新计算最新 observation 的 KV cache，prefill latency 从 O(N) 降到 O(1)。对 40+ 步的长链路 Agent，TTFT 可以从 3s 降到 300ms。

**建议二：Disaggregated Prefill/Decode 重叠 Tool Latency**

Long-horizon tool chaining 的瓶颈是串行依赖：

```
Turn N: LLM think → call tool → 等结果 → Turn N+1: LLM think → ...
                   ↑ 这 200ms 的 tool latency 是纯等待
```

如果推理引擎支持 disaggregated prefill/decode（NVIDIA Dynamo / llm-d），可以在等 tool 结果返回时**预先填充下一轮的 context**，把 tool latency 和 prefill latency 重叠：

```
Turn N:   LLM think → call tool ──→ 等结果 ──→ Turn N+1
                                     ↓ (并行)
                          Prefill 引擎：预填充下一轮 context
```

**面试话术**：

> "作为推理工程师，我关注的是长链路 Agent 的 prefill 膨胀问题。每轮 tool call 的结果注入 message buffer 后，下轮 prompt 更长，prefill 线性变慢。我的建议是两层：第一，用 prompt caching 把 memory + tool schema 固定在 cache 前缀，prefill 从 O(N) 降到 O(1)；第二，利用 disaggregated prefill/decode 架构，在等待 tool 结果时预先填充下一轮 context，重叠 tool latency 和 prefill latency。这两个优化可以把 40 步 Agent 的端到端延迟降低 40-60%。"

---

## 补充：Q5/Q6/Q7 细节修正

| 原回答 | 精确补充 |
|------|------|
| Q5 "Agent 在 Thought 阶段设计好顺序" | 实际是 runtime 判断依赖关系：第二个 call 的 args 引用了第一个 call 的 result → 串行化 |
| Q6 显式 no-op "流程统一" | 补充：no-op 还是 **hallucination 审计信号**——trace 中可以统计"调用了多少次 no-op vs 错误调用了其他 tool" |
| Q7 "三层隔离" | 正确但概念层，标准答案是：白名单命令集 + sandbox 执行限制 + 审计日志 |

---

## 巩固检查清单

- [ ] 能闭卷讲出 Toolformer 的自监督训练流程（插入→执行→loss 对比→保留/丢弃）
- [ ] 能正确解释"小模型不适用 tool annotation"是因为 capacity bottleneck 而非 latency
- [ ] 能区分 `run_shell(cmd)` 和 `git_status()` 的攻击面差异
- [ ] 能画出 prompt caching 固定 memory 前缀的架构图
- [ ] 能用推理工程师视角讲出 Agent 长链路的两个优化方向（caching + prefill/tool overlap）
