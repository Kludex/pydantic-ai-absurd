# agent-sessions

Durable session event log, brains, and `wake()` for Pydantic AI on Postgres only - no Redis, no message broker, no daemon. Layered on top of [`pydantic-ai-absurd`](../pydantic-ai-absurd).

## What you get

- `Session` - append-only event log keyed by UUID with a versioned payload schema
- `@brain("name")` - register a background agent handler that runs under Absurd
- `wake(session, "name", ...)` - idempotent trigger with a single-active-brain-per-session default
- `BrainContext.agent_run(agent, prompt)` - runs a Pydantic AI agent with checkpointing + session history
- Compaction primitives (snapshots + summarization hook), pg_notify integration (lossy hint)

## Install

```bash
uv add agent-sessions
```

Requires Postgres with the Absurd schema installed.

## Quickstart

```python
from agent_sessions import Session, brain, wake, create_worker
from pydantic_ai import Agent

planner = Agent("anthropic:claude-sonnet-4-6", name="planner")

@brain("planner")
async def planner_brain(ctx):
    result = await ctx.agent_run(planner)
    await ctx.post(result.output)
```

See the repository `README.md` for the full architecture story.
