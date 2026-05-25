# 构建 MCP 服务端——Python + TypeScript SDK

> 大多数 MCP 教程只展示 stdio 的 hello-world。一个真正的服务端要暴露 tools 外加 resources 和 prompts、处理能力协商、发出结构化错误，并且能在不同 SDK 之间一致工作。本课端到端构建一个笔记（notes）服务端：标准库 stdio 传输、JSON-RPC 分发、三个服务端原语，以及一种纯函数风格，当你升级到 Python SDK 的 FastMCP 或 TypeScript SDK 时可以直接接入。

**类型：** Build
**语言：** Python（标准库，stdio MCP 服务端）
**前置要求：** Phase 13 · 06（MCP 基础）
**时间：** 约 75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个分发循环，从 stdin 读取 JSON-RPC 消息，并将响应写入 stdout。
- 按照 JSON-RPC 2.0 规范和 MCP 的附加错误码发出结构化错误响应。
- 在不重写工具逻辑的前提下，将标准库实现升级到 FastMCP（Python SDK）或 TypeScript SDK。

## 问题

在你可以使用远程传输（Phase 13 · 09）或认证层（Phase 13 · 16）之前，你需要一个干净的本地服务端。本地意味着 stdio：服务端由客户端作为子进程启动，消息通过换行分隔的 stdin/stdout 流动。

2025-11-25 规范规定，stdio 消息编码为带有显式 `\n` 分隔符的 JSON 对象。这里不需要 SSE；SSE 是旧的远程模式，正在 2026 年年中被移除（Atlassian 的 Rovo MCP 服务端于 2026 年 6 月 30 日弃用；Keboola 于 2026 年 4 月 1 日弃用）。对于 stdio，每行一个 JSON 对象就是整个有线传输格式。

笔记服务端是一个很好的模型，因为它同时锻炼了全部三个服务端原语。Tools 做变更（`notes_create`）。Resources 暴露数据（`notes://{id}`）。Prompts 提供模板（`review_note`）。本课的结构可以推广到任何领域。

## 核心概念

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 输出任何不是 JSON-RPC 信封的内容。调试日志输出到 stderr。
- 每个 Request 必须匹配一个带有相同 `id` 的 Response。
- Notification 绝对不能回复。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的功能。客户端依赖能力集来控制功能开关。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个条目包含 `name`、`description`、`inputSchema`。`tools/call` 接受 `{name, arguments}`，返回 `{content: [blocks], isError: bool}`。

Content blocks 是类型化的。最常见的有：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形态。协议层面的错误（未知方法、错误参数）是 JSON-RPC 错误。工具层面的错误（调用合法但工具执行失败）以 `{content: [...], isError: true}` 返回。这让模型在其上下文中看到失败信息。

### 实现 resources

Resources 按设计是只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...` 或自定义 scheme 如 `notes://`。

当你将数据作为 resource 而非 tool 暴露时：

- 模型不"调用"它；客户端可以根据用户请求将其注入上下文。
- 订阅（Subscriptions）让服务端可以在资源变更时推送更新（Phase 13 · 10）。
- Phase 13 · 14 使用 `ui://` 扩展了这一点，用于交互式资源。

### 实现 prompts

Prompts 是带有命名参数的模板。宿主将其展示为斜杠命令。一个 `review_note` prompt 可能接受一个 `note_id` 参数，并生成一个多消息的提示词模板，客户端将其输入给它的模型。

### stdio 传输的微妙之处

- 换行分隔 JSON。没有带长度前缀的帧格式。
- 不要缓冲。每次写入后调用 `sys.stdout.flush()`。
- 客户端控制生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### Annotations

每个 tool 可以携带 `annotations` 描述其安全属性：

- `readOnlyHint: true`——纯读取，可以安全重试。
- `destructiveHint: true`——不可逆副作用；客户端应进行确认。
- `idempotentHint: true`——相同输入产生相同输出。
- `openWorldHint: true`——与外部系统交互。

客户端使用这些来决定 UX（确认对话框、状态指示器）和路由（Phase 13 · 17）。

### 升级路径

`code/main.py` 中的标准库服务端约 180 行。FastMCP（Python）将同样逻辑压缩为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等效形态。当你准备好时，升级路径是直接替换；核心概念（capabilities、dispatch、content blocks）保持不变。

## Use It

`code/main.py` 是一个完整的笔记 MCP 服务端，通过 stdio 传输，纯标准库。它处理 `initialize`、`tools/list`、三个工具的 `tools/call`（`notes_list`、`notes_search`、`notes_create`）、每个笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` prompt。你可以通过管道传入 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

需要关注的重点：

- 分发器是一个以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器返回一个 content blocks 列表，而非裸字符串。
- 当执行器抛出异常时，设置 `isError: true`。

## Ship It

本课产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该技能搭建一个 MCP 服务端脚手架，包含正确的 tools/resources/prompts 划分和 SDK 升级路径。

## 练习

1. 运行 `code/main.py`，用手工构造的 JSON-RPC 消息驱动它。先执行 `notes_create`，然后通过 `resources/read` 获取新笔记。

2. 添加一个 `notes_delete` 工具，带有 `annotations: {destructiveHint: true}`。验证客户端会弹出确认对话框（这需要一个真实的宿主；Claude Desktop 可行）。

3. 实现 `resources/subscribe`，使得每当笔记被修改时，服务端推送 `notifications/resources/updated`。添加一个 keepalive 任务。

4. 将服务端移植到 FastMCP。Python 文件应缩减到 80 行以下。有线行为必须完全一致；用相同的 JSON-RPC 测试框架验证。

5. 阅读规范的 `server/tools` 部分，识别出本课服务端中未实现的 tool 定义字段。（提示：有好几个；挑选一个并添加它。）

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|----------------|----------|
| MCP 服务端 | "暴露工具的那个东西" | 通过 stdio 或 HTTP 使用 MCP JSON-RPC 协议的进程 |
| stdio 传输 | "子进程模式" | 服务端由客户端启动为子进程；通过 stdin/stdout 通信 |
| Dispatcher | "方法路由器" | JSON-RPC 方法名到处理函数的映射 |
| Content block | "工具结果块" | tool 响应中 `content` 数组的类型化元素 |
| `isError` | "工具级失败" | 表示工具执行失败；区别于 JSON-RPC 错误 |
| Annotations | "安全提示" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | 基于装饰器的 MCP 协议高层框架 |
| Resource URI | "可寻址数据" | 识别 resource 的 `file://`、`db://` 或自定义 scheme |
| Prompt template | "斜杠命令摘要" | 服务端提供的带有参数占位符的模板，供宿主 UI 使用 |
| Capability declaration | "功能开关" | 在 `initialize` 中声明的逐原语标志 |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) ——参考 Python 实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) ——并行的 TS 实现
- [FastMCP — server framework](https://gofastmcp.com/) ——MCP 服务端的装饰器风格 Python API
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) ——使用任一 SDK 的端到端教程
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) ——tools/* 消息的完整参考
