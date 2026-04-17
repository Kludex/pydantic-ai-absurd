# agent-sessions

Durable session event log, brains, and `wake()` for Pydantic AI on Postgres only - no Redis, no message broker, no daemon. Layered on top of [`pydantic-ai-absurd`](../pydantic-ai-absurd).

## What you get

- `Session` - append-only event log keyed by UUID, serialized per-session via a Postgres advisory lock, with snapshots and a Pydantic AI adapter (`session.messages()` returns `list[ModelMessage]` ready for `agent.run(..., message_history=...)`).
- `Workflow` - orchestrator object whose `@workflow.brain("name")` decorator registers brain handlers and schedules them with Absurd at decoration time. `workflow.wake(session, "name", ...)` is the idempotent trigger; `workflow.run()` starts the worker loop.
- `BrainContext` - the object passed to every brain: `.agent_run(agent, prompt)`, `.post()`, `.post_status()`, `.wake("other")`, `.sleep(seconds)`, plus `.session`, `.input`, `.absurd_ctx`.
- `Session.listen(conninfo=...)` - async iterator over new events via `LISTEN/NOTIFY`; pairs with `events(after=N)` for lossless reconciliation.

## Install

```bash
uv add agent-sessions
```

Requires Postgres with the Absurd schema installed and migrations applied (`apply_migrations(pool)`).

## Quickstart

```python
from absurd_sdk import AsyncAbsurd
from agent_sessions import Session, Workflow, apply_migrations
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent

# `absurd` and `pool` come from your app bootstrap.
await apply_migrations(pool)
workflow = Workflow(absurd=absurd, pool=pool)

planner = AbsurdAgent(Agent("openai:gpt-5.2", name="planner"), absurd)

@workflow.brain("planner")
async def planner_brain(ctx):
    result = await ctx.agent_run(planner, "what should we do next?")
    await ctx.post(result.output)

# HTTP side: append a user message and wake the brain.
session = await Session.create(pool)
await session.append(kind="user_message", actor="user", payload={"content": "hi"})
await workflow.wake(session, "planner")

# Worker process: brains are registered at decoration time, just start the loop.
await workflow.run()
```

## Events and visibility

Every brain append lands in `session_events` with a `kind` (`user_message`, `assistant_message`, `tool_call`, `tool_result`, `status_update`, `brain_started/finished/failed`, `snapshot_created`), an `actor`, and a `visibility` of `public` or `internal`. `session.events(after=N, visibility=Visibility.public)` is the UI read pattern; `session.messages()` is the agent read pattern and filters to messages the LLM can consume.

## Concurrency and idempotency

`workflow.wake(session, name, concurrency=...)` defaults to `queue`, which takes a session-level advisory lock so only one brain per session runs at a time. `concurrency="parallel"` skips the lock.

Dedup is handled by Absurd's native `idempotency_key`: two wakes with the same `dedup_key` resolve to the same task without spawning twice. If you don't pass `dedup_key`, it's derived from `(session_id, brain_name, sha256(input))`.

Chains propagate `causation_id` so you can trace who woke whom. `max_wake_depth` (default 20) bounds runaway loops.

## See also

- Repository `README.md` for the architecture diagram and the Vercel AI + Starlette example.
- `pydantic-ai-absurd` for the underlying durable-execution adapter.
