# Reflexion：语言化强化学习 · 错题本

> 日期：2026-05-28 | 得分：11.5/16 | 趋势：↑ (ReWOO 10.5 → Reflexion 11.5)

---

## 错题一：Actor 角色定位偏差（Q1 · 1.0/2）

**扣分原因**：把 Actor 描述为"分析问题生成计划"，混淆了 Actor 和 ReWOO Planner。

**精确角色定义**：

```
Actor         : 生成 ReAct 风格轨迹（thought → action → observation）
                不是规划者，是执行者

Evaluator     : 输入轨迹，输出评分（标量/启发式/自我评估）

Self-Reflector: 输入失败轨迹 + 评估结果，输出自然语言反思

情景记忆       : 存储反思列表，预置到下一次试验的 prompt 中
```

**执行顺序**：
```
Actor 执行 → Evaluator 打分 → 如果失败 → Self-Reflector 写反思 → 存入情景记忆 → 下次试验的 Actor 能看到反思
```

---

## 错题二：Reflexion 优势是"样本效率"不是"信息密度"（Q3 · 1.0/2）

**扣分原因**：说"自然语言的信息密度比梯度高"——不正确。梯度的信息密度极高（精确指向最优方向）。

**正确理解**：

| 维度 | 梯度 RL | Reflexion |
|------|------|------|
| 学习单位 | 权重更新（浮点数） | 自然语言反思（一段话） |
| 需要试验次数 | 数千次 | 1-3 次 |
| 优势 | 精确、可微、可批量 | **样本效率极高** |
| Reflexion 的"权重"等价物 | — | 情景记忆中预置到 prompt 的反思文本 |

Reflexion 不是比梯度"更好"——它是在**不需要权重更新的场景下**用自然语言完成了相同效果的学习。

---

## 错题三：Memory Rot 是同一个概念（Q5 · 1.0/2）

**扣分原因**：先说"不是一个概念"，然后描述了一模一样的解决方案。

**正确认知**：

Lesson 07（Memory）和 Lesson 03（Reflexion）的 memory rot 是**同一个概念**——都是"旧信息堆积→检索质量下降→Agent 被过时信息误导"。

| 场景 | Memory Rot 表现 | 缓解方案 |
|------|------|------|
| Lesson 07 档案记忆 | 旧 facts 淹没新 facts | periodic consolidation |
| Lesson 03 情景记忆 | 旧反思过时/错误 | 定期压缩 / TTL / sleep-time 清理 agent |

**三种具体缓解措施**（来自文档）：
1. **定期压缩**：合并相似反思，删除被推翻的旧反思
2. **TTL 机制**：超过 N 次试验后自动丢弃（Exercise 2：10 次试验 TTL）
3. **Sleep-time 清理 agent**：独立 agent 在空闲时审核情景缓冲区

---

## 面试核心知识速查

### Reflexion 三角色

```
Actor ──执行──→ Evaluator ──评分──→ Self-Reflector ──反思──→ 情景记忆
  ↑                                                            │
  └──────────── 下次试验的 prompt 中预置反思 ←───────────────────┘
```

### 三种评估器

| 类型 | 信号 | 信号强度 | 适用场景 |
|------|------|:---:|------|
| Scalar | 二元通过/失败或数值分数 | 最强 | 有 ground truth（测试通过/失败） |
| Heuristic | 预定义失败特征 | 中等 | 卡住检测、步数超限 |
| Self-evaluated | LLM 自我评分 | 最弱 | 无 ground truth，需配合工具验证 |

### Reflexion 何时有效/无效

| 有效 | 无效 |
|------|------|
| 有明确的失败信号 | 第一次就成功 |
| 任务可复现（同类问题会再出现） | 外部原因（网络中断、工具损坏） |
| 有足够的行动预算 | 反思变成迷信（偶发不稳定运行的叙述） |

### 2026 生产实例

| 系统 | 存储位置 | 在线/离线 |
|------|------|:---:|
| Letta sleep-time | 内存 memory blocks | 离线 |
| Claude Code CLAUDE.md | 文件系统 .md | 混合（手动触发，自动加载） |
| pro-workflow /learn-rule | 外部 .md 文件 | 在线 |
| LangGraph reflection node | Graph state | 在线 |
| OpenAI Agents SDK | Session memory | 在线（需自定义 Guardrail） |

---

## 巩固检查清单

- [ ] 能闭卷说出 Reflexion 三角色的执行顺序和数据流
- [ ] 能区分 Reflexion 的优势是"样本效率"而非"信息密度"
- [ ] 能意识到 Lesson 03 和 Lesson 07 的 memory rot 是同一概念
- [ ] 能设计混合评估器策略（scalar + heuristic + self-eval）
- [ ] 能说出至少 3 个 2026 年生产中的 Reflexion 实例及它们在哪里存储情景记忆
