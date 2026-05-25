# MCP 基础——原语、生命周期、JSON-RPC 基础

> 在 MCP 出现之前，每个集成都是一次性的。Model Context Protocol，最初由 Anthropic 于 2024 年 11 月发布，现由 Linux Foundation 旗下的 Agentic AI Foundation 管理，标准化了发现和调用过程，使得任何客户端都可以与任何服务端通信。2025-11-25 版本的规范定义了六个原语（三个服务端原语，三个客户端原语）、三阶段生命周期和 JSON-RPC 2.0 有线传输格式。掌握了这些，本阶段 MCP 章节的其余部分就只是阅读理解而已。

**类型：** Learn
**语言：** Python（标准库，JSON-RPC 解析器）
**前置要求：** Phase 13 · 01 至 05（工具接口与函数调用）
**时间：** 约 45 分钟

## 学习目标

- 命名全部六个 MCP 原语（服务端：tools、resources、prompts；客户端：roots、sampling、elicitation），并为每个给出一个使用场景。
- 走通三阶段生命周期（initialize、operation、shutdown），并说明每个阶段由谁发送什么消息。
- 解析和发出 JSON-RPC 2.0 的 request、response 和 notification 信封。
- 解释 `initialize` 时的能力协商是什么，以及没有它会出什么问题。

## 问题

在 MCP 出现之前，每个使用工具的 Agent 都有自己的一套协议。Cursor 有一个形似 MCP 但不兼容的工具系统。Claude Desktop 推出了另一套。VS Code 的 Copilot 扩展有第三套。一个构建了"Postgres 查询"工具的团队需要为不同的宿主 API 将同一个工具编写三次。复用需要复制代码。

结果是各色一次性集成呈寒武纪大爆发之势，生态系统的进化速度因此触及天花板。

MCP 通过标准化有线传输格式解决了这个问题。单个 MCP 服务端可在每个 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月已有 300+ 客户端。SDK 月下载量达 1.1 亿。10,000 个以上的公开服务端。Linux Foundation 于 2025 年 12 月在新成立的 Agentic AI Foundation 下接管了管理工作。

本阶段使用的规范版本为 **2025-11-25**。该版本新增了异步 Tasks（SEP-1686）、URL 模式 elicitation（SEP-1036）、带 tools 的 sampling（SEP-1577）、增量范围同意（SEP-835）和 OAuth 2.1 资源指示符语义。Phase 13 · 09 至 16 涵盖这些扩展。本课止步于基础知识。

## 核心概念

### 三个服务端原语

1. **Tools。** 可调用的操作。与 Phase 13 · 01 相同的四步循环。
2. **Resources。** 暴露的数据。通过 URI 寻址的只读内容：`file:///path`、`db://query/...`、自定义 scheme。
3. **Prompts。** 可复用的模板。宿主 UI 中的斜杠命令；服务端提供模板，客户端填充参数。

### 三个客户端原语

4. **Roots。** 服务端被允许访问的 URI 集合。客户端声明它们；服务端遵守它们。
5. **Sampling。** 服务端请求客户端模型执行一次补全。使得服务端托管的 Agent 循环无需在服务端持有 API 密钥。
6. **Elicitation。** 服务端在执行过程中向客户端用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每一项能力都恰好属于这六个原语之一。Phase 13 · 10 至 14 深入涵盖每个原语。

### 有线传输格式：JSON-RPC 2.0

每条消息都是一个包含以下字段的 JSON 对象：

- Request：`{jsonrpc: "2.0", id, method, params}`。
- Response：`{jsonrpc: "2.0", id, result | error}`。
- Notification：`{jsonrpc: "2.0", method, params}`——无 `id`，不期望响应。

基础规范约有 15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务端到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：initialize。**

客户端发送 `initialize`，携带其 `capabilities` 和 `clientInfo`。服务端回复自己的 `capabilities`、`serverInfo` 和其支持的规范版本。当客户端消化完响应后，发送 `notifications/initialized`。从此刻起，双方都可以根据协商好的能力发送请求。

**阶段 2：operation。**

双向通信。客户端调用 `tools/list` 来发现工具，然后调用 `tools/call` 来执行。如果服务端声明了 sampling 能力，它可以发送 `sampling/createMessage`。当服务端的工具集发生变化时，它可以发送 `notifications/tools/list_changed`。当用户更改 root 范围时，客户端可以发送 `notifications/roots/list_changed`。

**阶段 3：shutdown。**

任意一方关闭传输通道。MCP 中没有结构化的关闭方法；传输层（stdio 或 Streamable HTTP，Phase 13 · 09）承载连接结束的信号。

### 能力协商

`initialize` 握手中的 `capabilities` 是合约。以下是服务端的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务端声明它能够发送 `tools/list_changed` 通知，并支持 `resources/subscribe`。客户端通过声明自己的 capabilities 来同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端未声明 `sampling`，服务端就不得调用 `sampling/createMessage`。对称地：如果服务端未声明 `resources.subscribe`，客户端就不得尝试订阅。

这正是防止生态分裂的机制。一个不支持 sampling 的客户端仍然是一个合法的 MCP 客户端；一个不调用 `sampling` 的服务端仍然是一个合法的 MCP 服务端。它们只是不使用这项功能而已。

### 结构化内容和错误形态

`tools/call` 返回一个 `content` 数组，包含类型化块：`text`、`image`、`resource`。Phase 13 · 14 将 MCP Apps（`ui://` 交互式 UI）添加到该列表中。

错误使用 JSON-RPC 错误码。规范定义的附加码：`-32002` "Resource not found"、`-32603` "Internal error"，以及作为 `error.data` 的 MCP 特定错误数据。

### 客户端能力 vs 工具调用细节

一个常见混淆：`capabilities.tools` 表示客户端是否支持工具列表变更通知。而客户端**是否**会调用特定工具，这是由其模型驱动的运行时选择，而非能力标志。能力标志是规范层面的合约。模型的选择是与之正交的。

### 为什么是 JSON-RPC 而非 REST？

JSON-RPC 2.0（2010）是一种轻量级的双向协议。REST 是客户端发起的。MCP 需要服务端发起的消息（sampling、notifications），因此 JSON-RPC 以其对称的请求/响应形态成为自然选择。JSON-RPC 还可以干净地组合在 stdio 和 WebSocket/Streamable HTTP 之上，无需重新发明 HTTP 的请求形态。

## Use It

`code/main.py` 提供了一个最小化的 JSON-RPC 2.0 解析器和发射器，然后手动走通 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真正的传输层；只有消息形态。对照"延伸阅读"中链接的规范来校验每个信封。

需要关注的重点：

- `initialize` 双向声明 capabilities；响应中包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回 `tools` 数组；每个条目有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应 `content` 是一个 `{type, text}` 块的数组。

## Ship It

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一份 pcap 风格的 MCP 客户端-服务端交互记录，该技能标注每条消息属于哪个原语、哪个生命周期阶段，以及它依赖哪个能力。

## 练习

1. 运行 `code/main.py`。识别能力协商发生的那一行，并描述如果服务端未声明 `tools.listChanged` 会有什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形态：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在一个长时间运行的 `tools/call` 执行期间发出该通知，并确认客户端处理器会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范——整份文档约 80 页。识别出大多数服务端**不需要**的那个能力标志。提示：与资源订阅相关。

4. 在纸上草拟一个假设性的"cron job"功能应该属于哪个原语。（提示：服务端希望客户端在指定时间调用它。目前六个原语中没有一个适合。）MCP 的 2026 路线图中有一个关于此的草案 SEP。

5. 从 GitHub 上一个开源 MCP 服务端解析一份会话日志。统计 request vs response vs notification 消息。计算生命周期消息与操作消息的比例。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|----------------|----------|
| MCP | "Model Context Protocol" | 模型到工具发现与调用的开放协议 |
| 服务端原语 | "服务端暴露什么" | tools（操作）、resources（数据）、prompts（模板） |
| 客户端原语 | "客户端让服务端使用什么" | roots（访问范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "有线传输格式" | 对称的 request/response/notification 信封 |
| `initialize` 握手 | "能力协商" | 第一条消息对；服务端和客户端声明各自支持的功能 |
| `tools/list` | "发现" | 客户端向服务端询问其当前工具集 |
| `tools/call` | "调用" | 客户端请求服务端以参数执行某个工具 |
| `notifications/*_changed` | "变更事件" | 服务端通知客户端其原语列表已变更 |
| Content block | "类型化结果" | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "Spec Evolution Proposal" | 命名的草案提案（例如 SEP-1686 针对异步 Tasks） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) ——权威规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) ——六原语思维模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) ——2024 年 11 月发布文章
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) ——一周年回顾与 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) ——SEP-1686、1036、1577、835 和 1724 摘要
