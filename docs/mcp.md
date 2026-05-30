---
icon: lucide/wrench
---

# Tools & MCP servers

Most useful agents call tools. So the natural question is: when my durable agent uses a tool, *is the tool call durable too?*

Yes. Both kinds of tool, your own function tools and MCP servers, are checkpointed by default.

## Function tools are checkpointed

If your agent has ordinary Python function tools, pydantic-ai-absurd wraps them so each call is a checkpoint.

```python
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent

inner = Agent("openai:gpt-5.2", name="helper")

@inner.tool_plain
def charge(customer_id: str, cents: int) -> str:
    return billing.charge(customer_id, cents)  # a real side effect

agent = AbsurdAgent(inner, absurd)  # `charge` is now checkpointed
```

When the model calls `charge`, the result is recorded in Postgres. If the worker crashes after the charge but before the run finishes, the replay does **not** call `charge` again, it returns the stored result. The customer is charged once.

This is the same guarantee model and MCP calls get, and it's why durable-by-default is the right behavior: the case that bites you is the tool with a side effect, and that's exactly the one you'd forget to protect.

!!! note "The return value must be JSON-serializable"
    A checkpointed tool's return value is stored in Postgres, so it has to be JSON-serializable (the same constraint a task's return value has). Return plain data, not live objects like an open connection.

!!! tip "Truly pure tools just pay a tiny write"
    A pure tool like `add(2, 3)` is checkpointed too, which costs one small Postgres write per call. That's almost always worth it for the once-only guarantee on the tools that *do* matter.

## MCP servers are checkpointed too

A call to an [MCP](https://modelcontextprotocol.io) server is a network round-trip to an external process, so checkpointing it matters even more: a replay shouldn't hit the server twice. When you give your agent an `MCPToolset`, the wrapping happens automatically:

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

> **Anything the agent calls during a run, the model, MCP servers, and your function tools, is checkpointed, so a crash resumes from the last completed call and every side effect runs once.**

The only thing that *isn't* automatically checkpointed is the plain Python you write in the task body around `agent.run()`. If that has a side effect that must run once, wrap it in `ctx.step` yourself (see [How durability works](durability.md#making-your-own-steps)).

## What gets checkpointed, at a glance

| Operation | Checkpointed? | Why |
| --- | --- | --- |
| Model request | :material-check: Yes | Expensive, external, non-deterministic |
| MCP tool call | :material-check: Yes | Network round-trip to another process |
| MCP tool listing & instructions | :material-check: Yes | Avoids re-querying the server on replay |
| Function tool call | :material-check: Yes | So tools with side effects run exactly once |
| Your own task code | :material-close: No | Wrap it in `ctx.step` if it must run once |

Next up: taking this to production, the **[two-process split, scaling, and the gotchas](deployment.md)**.
