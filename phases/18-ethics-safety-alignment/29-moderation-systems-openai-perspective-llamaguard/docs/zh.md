# 内容审核系统——OpenAI、Perspective、Llama Guard

> 生产环境中的内容审核系统将第 12-16 课中定义的安全策略操作化。OpenAI Moderation API：`omni-moderation-latest`（2024），基于 GPT-4o 构建，一次调用即可分类文本+图像；多语言测试集上的表现比上一版本好 42%；响应 schema 返回 13 个类别布尔值——harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用隐藏延迟；标记时显示占位响应。Llama Guard 3/4（第 16 课）：14 个 MLCommons 危害类别、Code Interpreter Abuse、8 种语言（v3）、多图像（v4）。Perspective API（Google Jigsaw）：在 LLM 作为审核器的浪潮之前就已存在的毒性评分系统；主要是单维度毒性评分，附带 severe-toxicity/insult/profanity 变体；是内容审核研究的基线。弃用通告：Azure Content Moderator 于 2024 年 2 月弃用，2027 年 2 月退役，由 Azure AI Content Safety 替代。

**类型：** Build
**语言：** Python（标准库，三层审核框架）
**前置要求：** Phase 18 · 16（Llama Guard / Garak / PyRIT）
**时间：** 约 60 分钟

## 学习目标

- 描述 OpenAI Moderation API 的类别分类体系，以及它与 Llama Guard 3 的 MLCommons 分类体系有何不同。
- 描述三层审核模式（输入、输出、自定义），并说出每层的一种失效模式。
- 描述 Perspective API 作为前 LLM 时代基线的地位，以及为什么它至今仍在研究中使用。
- 陈述 Azure 弃用时间线。

## 问题

第 12-16 课描述了攻击和防御工具体系。第 29 课涵盖了在用户接触产品的表层操作化这些防御的部署级审核系统。三层模式是 2026 年的默认配置。

## 核心概念

### OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o。一次调用分类文本+图像。对大多数开发者免费。

类别（响应 schema 中的 13 个布尔值）：
- harassment、harassment/threatening
- hate、hate/threatening
- self-harm、self-harm/intent、self-harm/instructions
- sexual、sexual/minors
- violence、violence/graphic
- illicit、illicit/violent

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余类别仅限文本。

在 `code/main.py` 的代码框架中，为了教学简洁性，我们将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别折叠到其顶级父类别中。生产代码应使用完整的 13 类别 schema。

多语言测试集上比上一代 moderation 端点好 42%。逐类别评分；应用程序自行设定阈值。

### Llama Guard 3/4

在第 16 课中涵盖。14 个 MLCommons 危害类别（组织方式与 OpenAI 的 13 个响应 schema 布尔值不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）为原生多模态，12B。

OpenAI 和 Llama Guard 的分类体系有重叠但也有分歧。OpenAI 将 "illicit" 作为一个广泛类别；Llama Guard 则将 "violent crimes" 和 "non-violent crimes" 分开。部署方根据自身策略与分类体系的契合度做出选择。

### Perspective API（Google Jigsaw）

在 LLM 作为审核器的浪潮之前（2020 年以前）就已存在的毒性评分系统。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。单维度主评分（TOXICITY）附带子维度变体。

被广泛用作内容审核研究基线，因为其 API 稳定、文档完善，并且拥有多年的校准数据积累。对于现代与 LLM 配合使用的场景，Llama Guard 或 OpenAI Moderation 通常是更合适的选择。

### 三层模式

1. **输入审核。** 在生成前对用户提示词进行分类。若被标记则拒绝。延迟：一次分类器调用。
2. **输出审核。** 在交付前对模型输出进行分类。若被标记则替换为拒绝响应。延迟：生成后一次分类器调用。
3. **自定义审核。** 领域特定规则（正则表达式、白名单、业务策略）。在输入或输出阶段运行。

三层按设计顺序执行：输入审核必须在生成之前完成，输出审核在生成之后运行。并行性体现在层内——同时对同一文本运行多个分类器（例如 OpenAI Moderation + Llama Guard + Perspective）可以隐藏各个分类器的延迟。作为可选优化，可以在输入审核完成且延迟 token-1 流式输出时显示占位响应（"请稍等，正在检查……"）。标记行为可配置：拒绝、净化、升级至人工审核。

### 失效模式

- **仅输入。** 无法捕获输出幻觉（第 12-14 课中的编码攻击可绕过输入分类器）。
- **仅输出。** 允许任何输入到达模型；增加成本；将内部推理暴露给攻击者。
- **仅自定义。** 跨类别不够鲁棒；正则表达式很脆弱。

分层是默认做法。双重保险。

### Azure 弃用

Azure Content Moderator：2024 年 2 月弃用，2027 年 2 月退役。由基于 LLM 并与 Azure OpenAI 集成的 Azure AI Content Safety 替代。这次迁移是 Azure 部署方在 2024-2027 年的一个实操项目。

### 在 Phase 18 中的位置

第 16 课在 red-team 语境下覆盖了审核工具体系。第 29 课覆盖部署级审核。第 30 课以当前的双重用途能力证据收尾。

## Use It

`code/main.py` 构建了一个三层审核框架：输入审核器（关键词 + 类别分数）、输出审核器（对输出进行同样的分类）、自定义审核器（领域规则）。你可以将输入送入并观察哪一层捕获了什么。

## Ship It

本课产出 `outputs/skill-moderation-stack.md`。给定一个部署场景，它推荐审核栈配置：输入层用哪个分类器、输出层用哪个、自定义规则有哪些、边界案例用什么裁判。

## 练习

1. 运行 `code/main.py`。将一个无害、边界和有害输入通过所有三层。报告每种输入是哪一层触发了标记。

2. 以 Perspective API 风格的毒性评分扩展框架，聚焦于一个特定类别。比较其阈值行为与类别评分的差异。

3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。识别三个无法干净映射的类别。

4. 为一款代码助手部署（例如 GitHub Copilot）设计审核栈。识别最相关和最不相关的类别，并提出自定义规则。

5. Azure Content Moderator 于 2027 年 2 月退役。规划一次向 Azure AI Content Safety 的迁移。识别迁移中风险最高的要素。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|----------------|----------|
| OpenAI Moderation | "omni-moderation-latest" | 基于 GPT-4o 的 13 类别（文本）分类器，具有部分多模态支持 |
| Perspective API | "Google Jigsaw 毒性评分" | 前 LLM 时代的毒性评分基线 |
| Llama Guard | "MLCommons 14 类别" | Meta 的危害分类器（v3：8B 文本，8 种语言；v4：12B 多模态） |
| 输入审核 | "生成前过滤器" | 模型调用前对用户提示词进行分类 |
| 输出审核 | "生成后过滤器" | 模型输出交付前对其进行分类 |
| 自定义审核 | "领域规则" | 部署特定规则（正则、白名单、策略） |
| 分层审核 | "全部三层" | 标准生产部署模式 |

## 延伸阅读

- [OpenAI Moderation API 文档](https://platform.openai.com/docs/api-reference/moderations) ——omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) ——Llama Guard 代码仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) ——毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) ——Azure 替代方案
