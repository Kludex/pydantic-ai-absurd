# Pydantic AI Absurd

<p align="center"><em>Durable execution for Pydantic AI agents, on Postgres alone.</em></p>

---

When you put an agent in production, something uncomfortable happens: it runs for a while.

A real agent call isn't one HTTP request. It's a model call, then a tool call, then *another* model call, maybe an MCP server in the middle - tens of seconds, sometimes minutes. And in those seconds, things go wrong. Your worker gets redeployed. The machine runs out of memory. A spot instance disappears. The process you were counting on is simply gone.

So what happens to the run? With most setups, it's lost. You start again from the beginning, you pay for every token again, and your user waits twice.

**Pydantic AI Absurd** makes that not happen. You call `agent.run()` inside a durable task, and every model call and MCP call is checkpointed into Postgres. If the worker dies halfway through, a new worker picks the task back up and *resumes from the last completed step* - no restart, no re-spent tokens.

It's the same idea as Pydantic AI's Temporal integration. The difference: no Temporal, no Redis, no broker, no daemon. Just the Postgres you already have.

## Install

```bash
uv add pydantic-ai-absurd
```

You'll need a Postgres database (that's where Absurd keeps its state) and a Pydantic AI `Agent`. That's it.

## Use it

You author a task, call the agent inside it, spawn it from one place, and run it from another:

```python
from absurd_sdk import AsyncAbsurd
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent

absurd = AsyncAbsurd("postgresql://localhost/absurd", queue_name="agents")
agent = AbsurdAgent(Agent("openai:gpt-5.2", name="analyst"), absurd)

# You write the task; the agent is a durable callable inside it.
@absurd.register_task(name="analyse")
async def analyse(params, ctx):
    result = await agent.run(params["prompt"])
    return {"output": result.output}

# Spawn from anywhere - it just writes to Postgres and returns immediately.
await absurd.spawn("analyse", {"prompt": "Analyse Q3 revenue"})

# Run the worker in a separate process - it claims the task and runs it.
await absurd.start_worker()
```

If the worker dies mid-run, Absurd makes the task claimable again. A new worker re-runs the handler, the checkpointed model and MCP calls return their cached results instantly, and the run continues from where it stopped. The user gets their answer, and you paid for each call once.

## The shape

Three things, and they only ever talk through Postgres:

- **You write the task.** It's a normal async function. The agent is one durable step inside it; you can branch, call the agent twice, do whatever you need.
- **`spawn` doesn't run the agent.** It records a request to run it and returns immediately, so the web request that triggered it stays fast. The slow work happens elsewhere, later, durably.
- **The worker does the work.** Run as many as you like - they coordinate through the database, each claiming different tasks. Scale them up and down freely; spawned tasks wait safely in Postgres until a worker is ready.

`spawn` and `start_worker` belong in **different processes**, often different containers. Register your tasks in the worker process (the one calling `start_worker()`); the process that only spawns just needs the database URL and the task name.

## Continue a conversation

A single `spawn` is one run. For a chat, carry the prior messages into the next run: `agent.run()` accepts `message_history`, and a finished run hands its messages back.

```python
from pydantic import TypeAdapter
from pydantic_ai import ModelMessage
from pydantic_core import to_jsonable_python

_history = TypeAdapter(list[ModelMessage])

@absurd.register_task(name="chat")
async def chat(params, ctx):
    history = _history.validate_python(params["message_history"]) if params.get("message_history") else None
    result = await agent.run(params["prompt"], message_history=history)
    return {"output": result.output, "all_messages": to_jsonable_python(result.all_messages())}
```

Store `all_messages` wherever your app keeps conversation state, and feed it back on the next turn. The run stays durable; the conversation is just data you carry forward.

## What gets checkpointed

The rule is simple: the expensive, external things are durable; your cheap, idempotent code is not.

- **Model requests** are checkpointed. The `ModelResponse` is cached and replayed, so a crash never re-calls the LLM for a step it already finished.
- **MCP tool calls** are checkpointed too - a call to an MCP server is a network round-trip, as expensive and external as a model call.
- **Plain function tools and your own task code** are *not* checkpointed - they're expected to be cheap and idempotent, so they just re-run on replay. If you have a side effect that must happen exactly once, wrap it in `ctx.step(...)` yourself.

## Documentation

Full docs - a step-by-step tutorial, the durability model in detail, tools and MCP, and a production guide - live at the [documentation site](https://kludex.github.io/pydantic-ai-absurd/).

## Develop

```bash
scripts/install   # uv sync
scripts/check     # ruff format --check + ruff check + mypy strict
scripts/test      # pytest + 100% coverage gate
```

Tests run against a real Postgres via `testcontainers`, so Docker must be up.

The example under `examples/` can be smoke-tested end to end against real OpenAI (kept out of CI, local use only):

```bash
OPENAI_API_KEY=... uv run pytest examples/tests/
```

The tests auto-skip when `OPENAI_API_KEY` isn't set.
