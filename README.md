# agent-workflow

Durable, crash-safe background agents for [Pydantic AI](https://github.com/pydantic/pydantic-ai) on Postgres alone - no Redis, no message broker, no daemon.

## Architecture

```mermaid
flowchart LR
    HTTP[HTTP handler] -->|wake| Absurd[(Postgres<br/>absurd.* schema)]
    Worker[Worker process] -->|claim| Absurd
    Worker -->|runs| Brain[@brain handler]
    Brain -->|agent_run| Agent[AbsurdAgent]
    Agent -->|checkpointed step| Absurd
    Agent -->|LLM / MCP call| LLM[Claude / OpenAI / MCP servers]
    Brain -->|append events| Session[(Postgres<br/>session_events)]
    Brain -->|wake| Absurd
    HTTP -->|read| Session
```

Two packages, built together:

- **`pydantic-ai-absurd`** wraps a Pydantic AI `Agent` so every model call and MCP tool call is checkpointed into [Absurd](https://github.com/earendil-works/absurd). A crashed worker replays from the checkpoint - no tokens re-spent.
- **`agent-sessions`** adds the session event log, a `Workflow` orchestrator whose `@workflow.brain(...)` decorator registers brain handlers, and an idempotent `workflow.wake(...)` that schedules brains as Absurd tasks. Default policy is one active brain per session; chains propagate `causation_id` for tracing.

## Use a brain

```python
from absurd_sdk import AsyncAbsurd
from agent_sessions import Session, Workflow
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent

workflow = Workflow(absurd=absurd, pool=pool)

planner = AbsurdAgent(Agent('anthropic:claude-sonnet-4-6', name='planner'), absurd, name='planner')

@workflow.brain('planner')
async def planner_brain(ctx):
    result = await ctx.agent_run(planner, 'what should we do next?')
    await ctx.post(result.output)
    if result.output.needs_analyst:
        await ctx.wake('analyst')

# HTTP handler
session = await Session.create(pool)
await session.append(kind='user_message', actor='user', payload={'content': 'hi'})
await workflow.wake(session, 'planner')

# Worker (separate process - brains are registered with Absurd the moment
# they're decorated, so just start the loop)
await workflow.run()
```

The worker survives restarts mid-run: when it comes back, Absurd replays from the last checkpoint and the brain continues where it left off.

## Develop

```bash
scripts/install   # uv sync --all-packages
scripts/check     # ruff format --check + ruff check + mypy strict
scripts/test      # pytest + 100% coverage gate (both packages)
```

Tests run against a real Postgres via `testcontainers`. Docker must be up.
