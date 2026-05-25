# Reflexion：语言化强化学习

> 基于梯度的 RL 需要数千次试验和一个 GPU 集群来修复一个失败模式。Reflexion（Shinn 等人，NeurIPS 2023）用自然语言做到了这一点：每次失败试验后，agent 写一条反思，将其存储在情景记忆中，并在下一次试验中以该记忆为条件进行推理。这就是 Letta 的 sleep-time compute、Claude Code 的 CLAUDE.md learnings 以及 pro-workflow 的 learn-rule 背后的模式。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** Phase 14 · 01（Agent Loop）、Phase 14 · 02（ReWOO）
**时间：** 约 60 分钟

## 学习目标

- 说出 Reflexion 的三个组成部分（Actor、Evaluator、Self-Reflector）以及情景记忆的作用。
- 实现一个基于标准库的 Reflexion 循环，包含二元评估器、反思缓冲区和全新的重新尝试。
- 针对给定任务，在标量、启发式和自我评估的反馈来源之间做出选择。
- 解释为什么语言化强化能捕捉到基于梯度的 RL 需要数千次试验才能修复的错误。

## 问题

一个 agent 任务失败了。在标准 RL 中，你需要再运行数千次试验，计算梯度，更新权重。昂贵、缓慢，而且大多数生产 agent 没有为每次失败投入训练预算的条件。

Reflexion（Shinn 等人，arXiv:2303.11366）提出了一个不同的问题：如果 agent 只是思考一下为什么失败，然后把那个想法放在 prompt 里再次尝试呢？没有权重更新。没有梯度。仅仅是在试验之间存储的自然语言。

结果：在 ALFWorld 上，它击败了 ReAct 和其他非微调基线。在 HotpotQA 上，它相比 ReAct 有所提升。在代码生成（HumanEval/MBPP）上，它当时达到了 state of the art。全程没有任何梯度步骤。

## 概念

### 三个组成部分

```
Actor         : 生成轨迹（ReAct 风格循环）
Evaluator     : 为轨迹打分——二元、启发式或自我评估
Self-Reflector: 对失败写出自然语言反思
```

再加一个数据结构：

```
情景记忆: 先前的反思列表，预置到下一次试验的 prompt 中
```

一次试验运行 Actor。Evaluator 为其打分。如果分数低，Self-Reflector 生成一条反思（"我选错了工具，因为我把问题误读为询问 X，而实际上是询问 Y"）。反思进入情景记忆。下一次试验重新开始，但可以看到反思。

### 三种评估器类型

1. **标量（Scalar）**——外部二元信号。ALFWorld 成功或失败。HumanEval 测试通过或失败。最简单、信号最强。
2. **启发式（Heuristic）**——预定义的失败特征。"如果 agent 连续两次产生相同的动作，标记为卡住。""如果轨迹超过 50 步，标记为低效。"
3. **自我评估（Self-evaluated）**——LLM 对自己的轨迹打分。在没有 ground truth 可用时需要。信号较弱；与工具接地验证（第 05 课——CRITIC）搭配效果更好。

2026 年的默认做法是混合使用：有标量时用标量，没有时用自我评估，启发式作为安全护栏。

### 为什么这具有泛化性

Reflexion 与其说是一个新算法，不如说是一种已命名的模式。几乎每个生产级"自愈"agent 都运行某种变体：

- Letta 的 sleep-time compute（第 08 课）：一个独立的 agent 反思过去的对话并写入 memory blocks。
- Claude Code 的 `CLAUDE.md` / "save memory" 模式：反思被捕获为 learnings，预置到未来的会话中。
- pro-workflow 的 `/learn-rule` 命令：纠正被捕获为显式规则。
- LangGraph 的 reflection 节点：一个节点对输出打分，并在需要时路由到 refine。

所有这些都源于同一个洞察：自然语言是一个足够丰富的媒介，可以在运行之间传递"我从失败中学到了什么"。

### 它何时有效，何时无效

Reflexion 在以下情况有效：

- 存在明确的失败信号（测试失败、工具错误、答案错误）。
- 任务类别是可复现的（同一类型的问题可以再次提出）。
- 反思有改进轨迹的空间（足够的行动预算）。

Reflexion 在以下情况无效：

- agent 在第一次尝试时就已经成功。
- 失败是外部原因（网络中断、工具损坏）——对"网络中断了"的反思对未来的运行没有帮助。
- 反思变成了迷信——存储关于一次偶发性不稳定运行的叙述。

2026 年的陷阱：记忆腐烂。反思会累积；其中一些已过时或错误；随着情景缓冲区增长，重新运行变得越来越慢。缓解措施：定期压缩（第 06 课）、反思的 TTL，或一个独立的 sleep-time 清理 agent（Letta）。

## 构建它

`code/main.py` 在一个玩具谜题上实现了 Reflexion：生成一个和为目标的 3 元素列表。Actor 发出候选列表；Evaluator 检查和；Self-Reflector 写一行关于哪里出错的说明。反思进入情景记忆以供下一次试验使用。

组件：

- `Actor`——一个脚本化策略，当看到反思时会改进。
- `Evaluator.binary()`——对目标和进行通过/失败判断。
- `SelfReflector`——生成对失败的一行诊断。
- `EpisodicMemory`——具有 TTL 语义的有界列表。

运行它：

```
python3 code/main.py
```

轨迹显示三次试验。试验 1 失败，存储一条反思，试验 2 看到反思并改进但仍然失败，试验 3 成功。与无反思的基线运行对比——它会一直卡在试验 1 的答案上。

## 使用它

LangGraph 将 reflection 作为一个节点模式提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 将情景缓冲区外部化为 markdown 文件。Letta 的 sleep-time compute 在空闲时间运行 Self-Reflector，使主 agent 保持延迟受限。OpenAI Agents SDK 不直接提供 Reflexion；你需要构建一个自定义 Guardrail，通过分数拒绝轨迹，并使用跨运行持久化的 memory `Session`。

## 交付它

`outputs/skill-reflexion-buffer.md` 创建并维护一个具有反思捕获、TTL 和去重功能的情景缓冲区。给定一个任务类别和一个失败，它输出的反思能真正帮助下一次试验（而不是泛泛的"要更小心"）。

## 练习

1. 从二元评估器切换到返回距离度量（距离目标多远）的标量评估器。它收敛更快吗？
2. 为反思添加 10 次试验的 TTL。超过这个时间点后，旧反思是有害还是有益？
3. 实现启发式评估器：如果相同的动作重复出现，标记试验为卡住。这如何与 Self-Reflector 交互？
4. 使用一个忽略反思的对抗性 Actor 运行 Reflexion。迫使 Actor 注意到反思的最小反思 prompt 工程是什么？
5. 阅读 Reflexion 论文第 4 节关于 AlfWorld 的内容。概念性地复现 130% 的成功率提升：与纯 ReAct 的关键差异是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Reflexion | "自我纠正" | Shinn 等人 2023——Actor、Evaluator、Self-Reflector 加上情景记忆 |
| Verbal reinforcement | "无需梯度的学习" | 自然语言反思预置到下一次试验的 prompt 中 |
| Episodic memory | "按任务分类的反思" | 针对一个任务类别的先前反思的有界缓冲区 |
| Scalar evaluator | "二元成功信号" | 来自 ground truth 的通过/失败或数值分数 |
| Heuristic evaluator | "基于模式的检测器" | 预定义的失败特征（如 stuck-loop、too-many-steps） |
| Self-evaluator | "对自己的轨迹做法官式 LLM 评估" | 当没有 ground truth 时的低信号 fallback——与工具接地验证搭配使用 |
| Memory rot | "陈旧的反思" | 情景缓冲区充满了过时条目；通过压缩/TTL 修复 |
| Sleep-time reflection | "异步自我反思" | 在热路径之外运行 Self-Reflector，使主 agent 保持快速 |

## 延伸阅读

- [Shinn 等人，Reflexion: Language Agents with Verbal Reinforcement Learning（arXiv:2303.11366）](https://arxiv.org/abs/2303.11366)——规范论文
- [Letta，Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute)——生产环境中的异步反思
- [Anthropic，Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)——将情景缓冲区作为上下文的一部分进行管理
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview)——reflection 节点模式
