# Red-Teaming：PAIR 与自动化攻击

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419)。PAIR——Prompt Automatic Iterative Refinement——是经典的自动化黑盒越狱方法。一个带有 red-team 系统提示词的攻击者 LLM 以迭代方式为攻击目标 LLM 生成越狱提示词，在其自身的对话历史中积累尝试和响应作为上下文反馈。PAIR 通常在 20 次查询内成功，比 GCG（Zou 等人提出的 token 级别梯度搜索）效率高出几个数量级，且不需要白盒访问权限。PAIR 如今已成为 JailbreakBench (arXiv:2404.01318) 和 HarmBench 中的标准基线，与 GCG、AutoDAN、TAP 和 Persuasive Adversarial Prompt 并列。

**类型：** Build
**语言：** Python（标准库，针对模拟目标的 mock PAIR 循环）
**前置要求：** Phase 18 · 01（指令遵循），Phase 14（Agent 工程）
**时间：** 约 75 分钟

## 学习目标

- 描述 PAIR 算法：攻击者系统提示词、迭代优化、上下文反馈。
- 解释为什么在目标为黑盒时，PAIR 严格优于 GCG。
- 列举另外四种自动化攻击基线（GCG、AutoDAN、TAP、PAP），并说明各自的显著特征。
- 描述 JailbreakBench 和 HarmBench 评估协议，以及各自语境下"攻击成功率"的含义。

## 问题

Red-teaming 曾经是一项人工活动。少数专家测试员构建对抗性提示词并追踪哪些有效。这无法规模化：攻击成功率需要统计样本，而且随着每次模型发布，攻击目标都在动态变化。PAIR 将 red-teaming 操作化为一个针对黑盒目标的优化问题。

## 核心概念

### PAIR 算法

输入：
- 目标 LLM T（我们正在攻击的模型）。
- 裁判 LLM J（打分响应是否为越狱成功）。
- 攻击者 LLM A（red-team 优化器）。
- 目标字符串 G："回应 [有害指令]"。
- 预算 K（通常为 20 次查询）。

循环，对于 k = 1..K：
1. 向 A 提供目标 G 以及迄今为止的（提示词, 响应）历史记录作为提示。
2. A 产出新提示词 p_k。
3. 将 p_k 提交给 T；接收响应 r_k。
4. J 根据目标对 (p_k, r_k) 打分。
5. 若分数 >= 阈值，则停止——成功找到越狱。
6. 否则，将 (p_k, r_k) 追加到 A 的历史记录中；继续。

实证结果（NeurIPS 2023）：对 GPT-3.5-turbo、Llama-2-7B-chat 的攻击成功率 >50%；平均成功所需查询次数在 10-20 范围内。

### 为什么 PAIR 更高效

GCG（Zou 等人，2023）通过梯度搜索对抗性 token 后缀；它需要白盒模型访问权限，且产生的后缀不可读。PAIR 是黑盒的，且产生自然语言攻击，可以在模型之间迁移。PAIR 的上下文反馈让攻击者能够从每次拒绝中学习；GCG 没有等效机制（每次新的 token 更新都需要重新获取之前的进展）。

### 相关自动化攻击

- **GCG（Zou 等人，2023，arXiv:2307.15043）。** Token 级别梯度搜索对抗性后缀。白盒，可迁移，产生不可读字符串。
- **AutoDAN（Liu 等人，2023）。** 在层次化目标引导下对提示词进行进化搜索。
- **TAP（Mehrotra 等人，2024）。** 带剪枝的攻击树——对多个 PAIR 风格的展开进行分支。
- **PAP（Zeng 等人，2024）。** Persuasive Adversarial Prompts——将人类说服技巧编码为提示词模板。

### JailbreakBench 和 HarmBench

两者（2024）标准化了评估：

- JailbreakBench (arXiv:2404.01318)。100 种有害行为，涵盖 10 个 OpenAI 政策类别。攻击成功率（ASR）是主要指标。需要一个裁判（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika 等人，2024）。510 种行为，涵盖 7 个类别，包含语义和功能性危害测试。在 33 个模型上比较了 18 种攻击方法。

ASR 通常在固定查询预算下报告。比较攻击需要在匹配的预算下进行；200 次查询下 90% 的 ASR 与 20 次查询下 85% 的 ASR 不可比较。

### 对 2026 年部署的意义

每个前沿实验室现在都会在发布前对生产模型运行 PAIR 和 TAP。ASR 轨迹出现在模型卡（第 26 课）和安全案例附录（第 18 课）中。这种攻击并非奇技淫巧——它是标准基础设施。

### 在 Phase 18 中的位置

第 12 课是自动化攻击的基础。第 13 课（Many-Shot Jailbreaking）是一种互补的长度利用攻击。第 14 课（ASCII Art / Visual）是一种编码攻击。第 15 课（Indirect Prompt Injection）是 2026 年的生产攻击面。第 16 课涵盖防御工具体系（Llama Guard、Garak、PyRIT）。

## Use It

`code/main.py` 构建了一个模拟 PAIR 循环。目标是一个模拟分类器，拒绝"明显"的有害提示词（关键词过滤）。攻击者是一个基于规则的优化器，尝试改写、角色扮演框架和编码。裁判对响应打分。你将观察到攻击者在关键词过滤下约 5-15 次迭代内成功，而在语义过滤下失败。

## Ship It

本课产出 `outputs/skill-attack-audit.md`。给定一份 red-team 评估报告，它审计：运行了哪些攻击（PAIR、GCG、TAP、AutoDAN、PAP）、各自的预算是多少、使用哪个裁判、针对哪个有害行为集（JailbreakBench、HarmBench、内部数据集）。

## 练习

1. 运行 `code/main.py`。测量三种内置攻击者策略的平均成功所需查询次数。说明每种策略利用了目标防御的哪个假设。

2. 实现第四种攻击者策略（例如，翻译为另一种语言、base64 编码）。报告针对关键词过滤目标和语义过滤目标的新平均成功所需查询次数。

3. 阅读 Chao 等人 2023 年论文的图 5（PAIR 与 GCG 对比）。描述两种 GCG 优于 PAIR 的场景——尽管 PAIR 有整体效率优势。

4. JailbreakBench 在固定目标集上报告 ASR。设计一个额外的指标来测量攻击多样性（成功提示词的方差）。解释为什么多样性对防御评估很重要。

5. TAP（Mehrotra，2024）通过分支+剪枝扩展了 PAIR。草拟一个对 `code/main.py` 的 TAP 风格扩展，并描述计算成本与成功率之间的权衡。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|----------------|----------|
| PAIR | "自动化越狱" | Prompt Automatic Iterative Refinement；攻击者 LLM + 裁判 LLM 循环 |
| GCG | "梯度越狱" | 白盒 token 级别梯度搜索对抗性后缀 |
| 攻击成功率 (ASR) | "k 次查询下的越狱百分比" | 主要指标；必须连同查询预算和裁判身份一起报告 |
| 裁判 LLM | "评分器" | 判断响应是否满足有害目标的 LLM |
| JailbreakBench | "标准化评估" | 带标签类别的标准化有害行为集 |
| HarmBench | "更广泛的基准" | 510 种行为，功能性 + 语义危害测试 |
| TAP | "攻击树" | 带分支+剪枝的 PAIR；更高计算成本下获得更好的 ASR |

## 延伸阅读

- [Chao 等人 — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) ——PAIR 论文，NeurIPS 2023
- [Zou 等人 — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) ——GCG 论文
- [Chao 等人 — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) ——标准化评估
- [Mazeika 等人 — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) ——更广泛的评估
