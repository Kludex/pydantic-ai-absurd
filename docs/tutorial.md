---
icon: lucide/graduation-cap
---

# Tutorial

This tutorial shows you, step by step, how to take a normal Pydantic AI agent and make a single run of it **survive a crash**.

We'll build it up one piece at a time. Each step adds exactly one new idea, shows you the code, and then explains *why*. By the end you'll have run an agent, killed its worker mid-flight, and watched it pick up exactly where it left off, without calling the model again.

!!! tip "Run it as you read"
    Every snippet here is real. If you have a Postgres handy and an API key set, you can paste these in and watch them work.

## What you'll need

- A Postgres database. Absurd stores its task state there.
- A Pydantic AI `Agent`, so an LLM provider key (we'll use `openai:gpt-5.2`).
- `pip install pydantic-ai-absurd`.

The first time you use Absurd it needs its schema installed and a queue created. You do this **once**. The schema is a SQL install you run against your database, the way you'd run any migration:

```bash
psql "postgresql://localhost/absurd" -f tests/fixtures/absurd.sql
```

The queue is created in code with `await absurd.create_queue()`, which you'll see in the full script in Step 4. If your first run greets you with `schema "absurd" does not exist` or `database "..." does not exist`, the [Troubleshooting](troubleshooting.md) page has the fix.

## Step 1: Wrap your agent

Start with an ordinary Pydantic AI agent. Then wrap it.

```python hl_lines="4 5"
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent

inner = Agent("openai:gpt-5.2", name="analyst")
agent = AbsurdAgent(inner, absurd)
```

That's the only change to your agent. `AbsurdAgent` keeps everything about `inner`, its model, its tools, its output type, but swaps the model (and any MCP tools) for versions that checkpoint each call.

!!! warning "The agent needs a name"
    The `name` isn't decoration, Absurd uses it as the prefix for every checkpoint step, so two agents with durable steps need two distinct names. Here it comes from the inner `Agent(..., name="analyst")`, and `AbsurdAgent` reuses it. If your inner agent has no name, pass one to `AbsurdAgent` directly: `AbsurdAgent(inner, absurd, name="analyst")`. Either way, if there's no name at all you'll get a clear error.

On its own, the wrapped agent does nothing special yet. The magic only happens when you call it *inside a task*. That's the next step.

## Step 2: Write a task

A "task" is just an async function you register with Absurd. You author it; the agent is a callable *inside* it.

```python hl_lines="3"
@absurd.register_task(name="analyse")
async def analyse(params, ctx):
    result = await agent.run(params["prompt"])
    return {"output": result.output}
```

A few things to notice:

- **You write the task.** This is the same shape as Pydantic AI's Temporal integration, you control the workflow, and the agent is one durable step within it. You can do other things in here too: branch, call the agent twice, log, whatever.
- `params` is whatever you pass when you spawn the task (more on that in a second). It's plain JSON.
- `ctx` is the Absurd task context. You usually don't touch it directly, the wrapped agent uses it under the hood to record checkpoints.
- The return value is the task's result, stored in Postgres. Keep it JSON-serializable.

## Step 3: Spawn it

Now, from anywhere (a FastAPI endpoint, a cron job, a "Generate report" button) you ask for the task to run:

```python
handle = await absurd.spawn("analyse", {"prompt": "Analyse Q3 revenue"})
print(handle["task_id"])
```

Here's the important part: **`spawn` doesn't run the agent.** It writes a row to Postgres saying "someone please run `analyse` with these params" and returns immediately. Your web request finishes in milliseconds. The actual work happens elsewhere, later, durably.

!!! tip "This is why your API stays fast"
    The slow, expensive agent run never blocks the request that triggered it. You spawn and move on.

## Step 4: Put it together and run it

Something has to actually *do* the work. Here's the whole thing in one runnable file. For trying things out, `work_batch` is the simplest way to drain tasks: it claims the ones that are waiting, runs them, and returns.

```python
import asyncio

from absurd_sdk import AsyncAbsurd
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent

absurd = AsyncAbsurd("postgresql://localhost/absurd", queue_name="agents")
agent = AbsurdAgent(Agent("openai:gpt-5.2", name="analyst"), absurd)


@absurd.register_task(name="analyse")
async def analyse(params, ctx):
    result = await agent.run(params["prompt"])
    return {"output": result.output}


async def main():
    await absurd.create_queue()
    await absurd.spawn("analyse", {"prompt": "Analyse Q3 revenue"})
    await absurd.work_batch(batch_size=1)


if __name__ == "__main__":
    asyncio.run(main())
```

`work_batch` claims your spawned task, runs the `analyse` function, stores the result, and *returns*, so your script finishes. That's exactly what you want while you're learning or testing.

!!! note "In production, use `start_worker`"
    `work_batch` does one pass and stops. A real worker process calls `await absurd.start_worker()` instead, which polls Postgres forever, runs tasks as they arrive, and resumes crashed runs. Same registration, same tasks: it just doesn't return. See [Running in production](deployment.md) for that shape.

!!! warning "Register your tasks where they run"
    The worker can only run tasks it knows about. The `@register_task` decorator must run **in the process that drains tasks** (the one calling `work_batch` or `start_worker`). The process that *spawns* doesn't need it: it only writes a task name and params to the database.

That's the full loop. Spawn from one place, run from another, talk only through Postgres.

## Step 5: Get the result

`spawn` hands you a task id, and `fetch_task_result` looks the result up once the task has run. Keep the id from `spawn` and fetch after `work_batch`:

```python
async def main():
    await absurd.create_queue()
    handle = await absurd.spawn("analyse", {"prompt": "Analyse Q3 revenue"})
    await absurd.work_batch(batch_size=1)

    result = await absurd.fetch_task_result(handle["task_id"])
    if result is not None and result.state == "completed":
        print(result.result["output"])
```

In a real app you'd typically have the worker write the result somewhere your users can see: a row in your own table, a webhook, a websocket push. `fetch_task_result` is the simple polling version.

## Step 6: The payoff, crash and resume

Here's the whole reason we did any of this.

Imagine the `analyse` agent makes two model calls (say, it uses a tool in between). The worker finishes the **first** model call, records its checkpoint... and then the machine dies.

What happens?

1. Absurd notices the task didn't finish and makes it claimable again.
2. A new worker claims it and runs `analyse` from the top.
3. `agent.run()` reaches the first model call, but that one is already checkpointed. Instead of calling the LLM again, it returns the **cached** response instantly.
4. Execution continues to the second model call, which *hasn't* run yet, and does it for real.
5. The task completes.

The user got their answer. You paid for each model call **once**. The crash cost you nothing but a few seconds.

!!! note "You didn't write any of that"
    No retry logic, no "where did I leave off" bookkeeping, no manual state. You wrote a normal function that calls `agent.run()`. The resume behavior comes from the checkpoints, for free.

## Recap

You went from a plain agent to a durable one in five small moves:

- [x] Wrap the agent with `AbsurdAgent`
- [x] Write a task with `@absurd.register_task`
- [x] `spawn` it from your app
- [x] Drain it with `work_batch` (or `start_worker` for a long-running worker)
- [x] Let crashes resume instead of restart

Now that you've *seen* it work, the next page explains exactly **[how durability works](durability.md)** under the hood, what counts as a checkpoint, what doesn't, and the one surprise to watch out for.
