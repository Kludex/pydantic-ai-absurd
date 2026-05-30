---
icon: lucide/wrench
---

# Tools & MCP servers

Most useful agents call tools. So the natural question is: when my durable agent uses a tool, *is the tool call durable too?*

The answer depends on what kind of tool it is, and pydantic-ai-absurd draws a deliberate line between two cases. Let's look at both.

## Plain function tools pass through

If your agent has ordinary Python function tools, pydantic-ai-absurd leaves them completely alone.

```python
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent

inner = Agent("openai:gpt-5.2", name="helper")

@inner.tool_plain
def add(a: int, b: int) -> int:
    return a + b

agent = AbsurdAgent(inner, absurd)  # `add` is untouched
```

The `add` tool is **not** wrapped in a checkpoint. When a task replays, `add` runs again like any other plain Python.

Why? Because that's the right default for function tools. They're expected to be **cheap and idempotent**, a calculation, a lookup, a pure transformation. Re-running `add(2, 3)` after a crash costs nothing and changes nothing. Wrapping it in a checkpoint would add Postgres round-trips for no benefit.

!!! warning "If a function tool has a real side effect"
    A function tool that charges a credit card, sends an email, or writes a row is *not* idempotent, and it's not checkpointed, so a replay would do it twice. For those, do the side effect through `ctx.step(...)` inside your task (see [How durability works](durability.md#making-your-own-steps)), or make the operation idempotent on your end.

## MCP servers get wrapped

[MCP](https://modelcontextprotocol.io) servers are a different story. A call to an MCP server is a network round-trip to an external process, it's exactly as expensive and external as a model call. So pydantic-ai-absurd **does** checkpoint those.

When you give your agent an `MCPToolset`, the wrapping happens automatically:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai_absurd import AbsurdAgent

toolset = MCPToolset("https://example.com/mcp")
inner = Agent("openai:gpt-5.2", name="researcher", toolsets=[toolset])

agent = AbsurdAgent(inner, absurd)
```

`MCPToolset` is Pydantic AI's unified way to talk to an MCP server, over HTTP, stdio, or an in-process server. You pass it a URL, a script path, or a server instance. When `AbsurdAgent` wraps the agent, it finds that `MCPToolset` and replaces it with a durable `AbsurdMCPToolset` automatically, you don't do anything.

Now every tool call to that MCP server is a checkpoint. Crash mid-run, and on replay the tool result comes back from Postgres instead of hitting the server again.

!!! note "Listing tools is checkpointed too"
    Not just calls, discovering *which* tools the server offers (`get_tools`) and its instructions are also checkpointed, so replay doesn't re-query the server for things it already learned.

## Caching tool definitions

An MCP server's list of tools rarely changes during a single run. So `AbsurdMCPToolset` caches it: it asks the server which tools exist once, then reuses that within the run.

This follows the wrapped toolset's `cache_tools` setting:

=== "Cached (default)"

    ```python
    toolset = MCPToolset("https://example.com/mcp")
    # cache_tools=True by default: tools listed once, reused for the run
    ```

    Good for almost every server. One fewer round-trip per run.

=== "Disabled"

    ```python
    toolset = MCPToolset("https://example.com/mcp", cache_tools=False)
    # re-fetches the tool list: use this if the server changes its tools mid-run
    ```

    Reach for this only if your server emits `tools/list_changed` notifications and you genuinely need the live list during a run.

## The rule of thumb

You can hold the whole thing in one sentence:

> **MCP calls are durable; plain function tools are not, because MCP calls are expensive and external, and function tools are meant to be cheap and idempotent.**

If a function tool *isn't* cheap and idempotent, that's your signal to either make it idempotent or move its side effect into an explicit `ctx.step`.

## What gets checkpointed, at a glance

| Operation | Checkpointed? | Why |
| --- | --- | --- |
| Model request | :material-check: Yes | Expensive, external, non-deterministic |
| MCP tool call | :material-check: Yes | Network round-trip to another process |
| MCP tool listing & instructions | :material-check: Yes | Avoids re-querying the server on replay |
| Plain function tool | :material-close: No | Expected to be cheap and idempotent |
| Your own task code | :material-close: No | Wrap it in `ctx.step` if it must run once |

Next up: taking this to production, the **[two-process split, scaling, and the gotchas](deployment.md)**.
