# A2A——Agent-to-Agent 协议

> MCP 是 agent-to-tool。A2A（Agent2Agent）是 agent-to-agent——一个开放协议，用于让构建在不同框架上的不透明 Agent 彼此协作。由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux Foundation，2026 年 4 月达到 v1.0，拥有 150+ 支持者，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP，并新增了 AP2 支付扩展。本课讲解 Agent Card、Task 生命周期和两种传输绑定。

**类型：** Build
**语言：** Python（标准库，Agent Card + Task 测试框架）
**前置要求：** Phase 13 · 06（MCP 基础），Phase 13 · 08（MCP 客户端）
**时间：** 约 75 分钟

## 学习目标

- 区分 agent-to-tool（MCP）和 agent-to-agent（A2A）的使用场景。
- 在 `/.well-known/agent.json` 发布 Agent Card，含技能和端点元数据。
- 走通 Task 生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带 Parts（text、file、data）的 Messages 以及作为输出的 Artifacts。

## 问题

一个客服 Agent 需要将报告撰写任务委派给一个专业的写作 Agent。A2A 出现之前的选项：

- 自定义 REST API。可行，但每一对组合都是一次性工作。
- 共享代码库。要求两个 Agent 运行相同的框架。
- MCP。不适用：MCP 用于调用工具，而非让两个 Agent 在保留各自不透明内部推理的前提下协作。

A2A 填补了这一空白。它将交互建模为一个 Agent 向另一个 Agent 发送 Task，具有生命周期、消息和产出物。被调用 Agent 的内部状态保持不透明——调用者只能看到任务状态转换和最终输出。

A2A 是"让来自不同框架的 Agent 相互对话"的协议。它不替代 MCP；两者是互补的。

## 核心概念

### Agent Card

每个 A2A 兼容的 Agent 在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现机制基于 URL：获取卡片，获取 A2A 端点 URL，枚举技能。

### Signed Agent Cards（AP2）

AP2 扩展（2025 年 9 月）为 Agent Card 增加了密码学签名。发布者使用 JWT 对其自己的卡片签名；消费者进行验证。防止冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working（通过消息循环）
```

客户端通过 `tasks/send` 发起。被调用 Agent 在各状态之间转换；客户端通过 SSE 订阅状态更新或轮询。

### Messages 和 Parts

一条 Message 携带一个或多个 Parts：

- `text`——纯文本内容。
- `file`——带 mimeType 的 base64 二进制块。
- `data`——类型化 JSON payload（面向被调用 Agent 的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

输出是 Artifacts，而非裸字符串。一个 Artifact 是一个命名的、类型化的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以以块（chunks）形式流式传输。调用者进行累积。

### 两种传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，POST 用于请求，可选的 SSE 用于流式传输。默认绑定。
2. **gRPC。** 适用于 gRPC 为原生方案的企业环境。

两种绑定承载相同的逻辑消息形态。

### 不透明性保留

一项关键设计原则：被调用 Agent 的内部状态是不透明的。调用者只能看到任务状态和 artifacts。被调用 Agent 的思维链、其工具调用、其子 Agent 委派——全部不可见。这与 MCP 不同，MCP 中的工具调用是透明的。

理由：A2A 使竞争对手可以在不暴露内部实现的情况下进行协作。A2A 可以"调用这个客服 Agent"，而调用者无需了解该 Agent 如何实现该服务。

### 时间线

- **2025-04-09。** Google 发布 A2A。
- **2025-06-23。** 捐赠给 Linux Foundation。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）发布。
- **2026-04。** v1.0 发布，150+ 支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|-----------|-----|-----|
| 使用场景 | Agent-to-tool | Agent-to-agent |
| 不透明性 | 透明的工具调用 | 不透明的内部推理 |
| 典型调用者 | Agent runtime | 另一个 Agent |
| 状态 | 工具调用结果 | 带生命周期的 Task |
| 授权 | OAuth 2.1（Phase 13 · 16） | JWT 签名 Agent Cards（AP2） |
| 传输 | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想调用一个特定工具时使用 MCP。当你想将整个任务委派给另一个 Agent 时使用 A2A。许多生产系统同时使用两者：Agent 使用 MCP 作为工具层，使用 A2A 作为协作层。

## Use It

`code/main.py` 实现了一个最小化 A2A 测试框架：一个研究 Agent 发布其卡片，一个写作 Agent 接收一个 `tasks/send`，消息中包含一个 PDF 和文本指令，经历 working → input_required → working → completed 状态转换，并返回一个文本 artifact。全部使用标准库；使用内存传输来聚焦于消息形态。

需要关注的重点：

- Agent Card JSON 形态。
- Task ID 分配和状态转换。
- 带混合类型 parts 的 Messages。
- 任务执行中期的 input-required 分支。
- 完成时返回的 Artifact。

## Ship It

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个新的、应可被其他 Agent 调用的 Agent，该技能生成 Agent Card JSON、技能 schema 和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的 Task 生命周期，包括被调用 Agent 请求澄清的 input-required 暂停。

2. 添加 Signed Agent Card。使用 HMAC 对卡片的标准 JSON 进行签名。编写验证器，并确认它对被篡改的卡片验证失败。

3. 实现任务流式传输：写作 Agent 通过 SSE 发出三个增量 artifact chunks，调用者累积它们。

4. 设计一个包装 MCP 服务端的 A2A Agent。将每个 MCP tool 映射为一个 A2A skill。记录权衡——什么不透明性被丧失了？

5. 阅读 A2A v1.0 公告，识别截至 2026 年 4 月尚未被任何框架实现的那个功能。（提示：与多跳任务委派相关。）

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|----------------|----------|
| A2A | "Agent-to-Agent 协议" | 用于不透明 Agent 协作的开放协议 |
| Agent Card | "`.well-known/agent.json`" | 描述 Agent 技能和端点的已发布元数据 |
| Skill | "一个可调用单元" | Agent 支持的一个命名操作（类比 MCP tool） |
| Task | "委派单元" | 具有生命周期和最终 artifact 的工作项 |
| Message | "Task 输入" | 携带 Parts（text、file、data） |
| Part | "类型化块" | 消息中的 `text` / `file` / `data` 元素 |
| Artifact | "Task 输出" | 完成时返回的命名、类型化输出 |
| AP2 | "Agent Payments Protocol" | 用于信任和支付的 Signed Agent Cards 扩展 |
| Opacity | "黑盒协作" | 被调用 Agent 的内部细节对调用者隐藏 |
| Input-required | "Task 暂停" | Agent 需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) ——权威 A2A 规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) ——参考实现和 SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) ——2025 年 6 月治理移交
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) ——路线图与合作伙伴势头
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) ——v1.0 发布说明和向后兼容性指导
