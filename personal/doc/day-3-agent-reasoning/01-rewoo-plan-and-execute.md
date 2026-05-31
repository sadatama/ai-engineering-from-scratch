# ReWOO 与 Plan-and-Execute · 错题本

> 日期：2026-05-27 | 得分：10.5/16 | 及格线：10/16
> 状态：压线及格，一个模式持续两个 Day 需根治

---

## 错题一：简单任务过度设计（Q6 · 1.0/2）

**扣分原因**："查天气"这种单次工具调用，选 ReWOO 是杀鸡用牛刀。

**正确判断**：

| 场景 | 正确模式 | 理由 |
|------|------|------|
| 单次 tool call "查天气" | **ReAct** | Anthropic 原则：最简单的模式。1 step 不需要规划 |
| 多步 "查财报+计算差值" | Plan-Execute | 有多个独立步骤，可以预先规划 |
| 40+ 步部署+错误处理 | Plan-Act | 长周期、环境不确定、需要审视和调整 |

**面试话术**：

> "模式选择的第一原则是 Anstropic 2024 指导的'从最简单的开始'。单次 tool call 用 ReWOO 是在做无用功——Planner 产生一个只有 1 个节点的 DAG，Solver 只需要把 worker 结果拼回去。这和直接 ReAct 一步做完没有任何区别。工程权衡就是知道什么时候不用复杂方案。"

**与 Day 2 同模式关联**：昨天 Advanced RAG Q8 也是把精确匹配和语义模糊搞反——两个错误都指向同一个根因：**不是每种场景都需要最复杂的方案**。

---

## 错题二：Replaner Token 开销估算模糊（Q7 · 1.0/2）

**扣分原因**：只说了"增加一段思考与 plan 的 token"但没有量化。

**精确数据**：

| 模式 | LLM 调用次数 | 相对 ReWOO |
|------|:---:|:---:|
| ReWOO | 2 次（1 plan + 1 solve） | 基准 |
| Plan-Execute | 2 + N_replan 次（每次 replan 是 1 次 LLM 调用） | +50% 起 |
| ReAct | N 次（每步 1 次） | 不可比 |

**面试话术**：

> "Replanner 的代价是每次触发额外 1 次 LLM 调用。相比 ReWOO 的 2 次调用，哪怕只触发 1 次 replan 就是 50% 的 LLM 开销增加。所以 Plan-Execute 只在任务足够复杂、静态计划可能失效时才值得。"

---

## 错题三：ReAct 角色映射不精确（Q1 · 1.5/2）

**扣分原因**：把 observation 映射为 Solver。

**精确映射**：

```
ReAct 的 LLM 承担了两个角色：
  thought → 对应 Planner（决定下一步做什么）
  最终 action: finish(answer) → 对应 Solver（整合所有信息给出答案）

Worker（工具调用）在 ReAct 和 ReWOO 中都不是 LLM 的角色。
Observation 是工具输出，不是 LLM 角色——它在任何模式中都是一样的。
```

---

## ReWOO vs ReAct 核心对比（面试速查）

| 维度 | ReAct | ReWOO |
|------|------|------|
| LLM 调用次数 | N 次（每步 1 次） | 2 次（1 plan + 1 solve） |
| Token 增长 | O(n²) 二次增长 | 固定，不随步数增长 |
| 灵活性 | 每一步都能根据 observation 调整 | 计划是静态的，无法中途调整 |
| 失败处理 | 必须从错误 observation 中自恢复 | 错误隔离到单节点，Solver 优雅降级 |
| Planner 蒸馏 | 不可蒸馏（每步都依赖上一步结果） | 可蒸馏（输入固定） |
| 适用场景 | 环境不确定、短任务 | 结构清晰、工具已知 |

---

## 巩固检查清单

- [ ] 能闭卷画出 ReWOO 三角色数据流图（Planner → Workers → Solver）
- [ ] 能解释为什么 ReAct token 增长是 O(n²) 而 ReWOO 是 O(1)
- [ ] 能准确判断单次 tool call / 多步结构化 / 长周期动态三种场景的模式选择
- [ ] 能说出 Planner 蒸馏的关键洞察：Planner 输入固定、看不到 observation
- [ ] 能对比 Replanner 和纯 ReWOO 的 LLM 调用次数差异（至少 +50%）
