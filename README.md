# pydantic-ai-absurd

Run a Pydantic AI agent durably on Postgres alone - no Redis, no broker, no daemon. Call `agent.run(...)` inside an [Absurd](https://github.com/earendil-works/absurd) task and every model call and MCP call is checkpointed; a worker crash mid-run resumes from the last completed step instead of restarting, without re-spending tokens.

It's the Postgres-only analogue of Pydantic AI's Temporal integration: you author the task, and the agent is a durable callable inside it.

## Install

```bash
uv add pydantic-ai-absurd
```

Requires Postgres (for Absurd) and a Pydantic AI `Agent`.

## Use it

Wrap the agent, write an Absurd task that calls `agent.run(...)`, and spawn the task:

```python
from absurd_sdk import AsyncAbsurd
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent

absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
agent = AbsurdAgent(Agent('openai:gpt-5.2', name='analyst'), absurd, name='analyst')

@absurd.register_task(name='analyse')
async def analyse(params, ctx):
    result = await agent.run(params['prompt'])
    return {'output': result.output}

# Client side (any process): enqueue a durable run, returns immediately.
await absurd.spawn('analyse', {'prompt': 'analyse Q3 revenue'})

# Worker side (separate process / container): claim and run durably.
await absurd.start_worker()
```

If the worker dies mid-run, Absurd re-enqueues the task; a new worker re-runs the handler, the checkpointed model/MCP calls return their cached results, and the run continues from where it stopped.

### Two processes

`spawn` only writes a row to Postgres, so the producer and the worker are fully decoupled - run them in separate processes or containers. The task name (`'analyse'`) must be registered in **the worker process** (the one that calls `start_worker()`); the producer just needs the DSN and the name.

### Continue a conversation

Pass a serialized `message_history` through the task params and hand it to `agent.run(...)` so a follow-up turn continues the conversation:

```python
from pydantic import TypeAdapter
from pydantic_ai import ModelMessage
from pydantic_core import to_jsonable_python

_history = TypeAdapter(list[ModelMessage])

@absurd.register_task(name='chat')
async def chat(params, ctx):
    history = _history.validate_python(params['message_history']) if params.get('message_history') else None
    result = await agent.run(params['prompt'], message_history=history)
    return {'output': result.output, 'all_messages': to_jsonable_python(result.all_messages())}
```

Persist `all_messages` wherever your app keeps conversation state and feed it back on the next turn.

## What gets checkpointed

- **Model requests** (`AbsurdModel`) - each `request()` / `request_stream()` is a `ctx.step(...)`; the `ModelResponse` is cached and replayed.
- **MCP / FastMCP tool calls** (`AbsurdMCPServer`, `AbsurdFastMCPToolset`) - each call is a step.
- **Plain function toolsets** are left untouched - their Python side effects are expected to be idempotent and cheap to re-run. Wrap anything non-idempotent in `ctx.step(...)` yourself.

## Develop

```bash
scripts/install   # uv sync
scripts/check     # ruff format --check + ruff check + mypy strict
scripts/test      # pytest + 100% coverage gate
```

Tests run against a real Postgres via `testcontainers`. Docker must be up.

The example under `examples/` can be smoke-tested end-to-end against real OpenAI (kept out of CI - local use only):

```bash
OPENAI_API_KEY=... uv run pytest examples/tests/
```

The tests auto-skip when `OPENAI_API_KEY` isn't set.
