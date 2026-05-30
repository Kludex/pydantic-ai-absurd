# pydantic-ai-absurd

Run a Pydantic AI agent as a durable [Absurd](https://github.com/earendil-works/absurd) workflow on Postgres alone - no Redis, no broker, no daemon. The whole `agent.run()` becomes one durable task; every model call and MCP call inside it is checkpointed, so a worker crash resumes from the last completed step instead of restarting (and without re-spending tokens).

This is the Postgres-only analogue of Pydantic AI's Temporal integration: the run is the workflow.

## Install

```bash
uv add pydantic-ai-absurd
```

Requires Postgres (for Absurd) and a Pydantic AI `Agent`.

## Run as a durable workflow

Pass `register_task=True` to register the whole run as an Absurd task named `<name>.run`, then spawn it:

```python
from absurd_sdk import AsyncAbsurd
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent

absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
inner = Agent('anthropic:claude-sonnet-4-6', name='analyst')
agent = AbsurdAgent(inner, absurd, name='analyst', register_task=True)

# HTTP side: enqueue a durable run. Returns immediately with a task id.
handle = await absurd.spawn('analyst.run', {'prompt': 'analyse Q3 revenue'})

# Worker side (separate process): claim and run it durably.
await absurd.start_worker()
```

If the worker dies mid-run, Absurd re-enqueues the task; a new worker replays the checkpointed steps and continues from where it stopped.

### Continue a conversation

The task accepts a serialized `message_history`, so a follow-up turn resumes the conversation rather than starting blank:

```python
from pydantic_core import to_jsonable_python

await absurd.spawn('analyst.run', {
    'prompt': 'and how does that compare to Q2?',
    'message_history': to_jsonable_python(previous_result.all_messages()),
})
```

The task result is `{'output': ..., 'all_messages': ...}` (both JSON) - persist `all_messages` wherever your app keeps conversation state and feed it back on the next turn.

## Use as a sub-component

Leave `register_task=False` (the default) when the agent isn't itself a task - e.g. you call `await agent.run(...)` from inside another Absurd task. You still get per-call checkpointing (the model and MCP calls replay from cache on retry); you just don't register a top-level `<name>.run` task nobody spawns.

```python
agent = AbsurdAgent(inner, absurd, name='analyst')  # no task registered

async def my_task(params, ctx):
    result = await agent.run(params['prompt'])  # checkpointed model/MCP calls
    ...
```

## What gets checkpointed

- **Model requests** (`AbsurdModel`) - each `request()` / `request_stream()` is a `ctx.step(...)`; the `ModelResponse` is cached and replayed.
- **MCP / FastMCP tool calls** (`AbsurdMCPServer`, `AbsurdFastMCPToolset`) - each call is a step.
- **Plain function toolsets** are left untouched - their Python side effects are expected to be idempotent and cheap to re-run. Wrap anything non-idempotent in `ctx.step(...)` yourself.
