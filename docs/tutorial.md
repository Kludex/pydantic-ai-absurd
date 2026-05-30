---
icon: lucide/graduation-cap
---

# Tutorial - User Guide

This tutorial shows you, step by step, how to take a normal Pydantic AI agent and make a single run of it **survive a crash**.

We'll build it up one piece at a time. Each step adds exactly one new idea, shows you the code, and then explains *why*. By the end you'll have run an agent, killed its worker mid-flight, and watched it pick up exactly where it left off - without calling the model again.

!!! tip "Run it as you read"
    Every snippet here is real. If you have a Postgres handy and an API key set, you can paste these in and watch them work.

## What you'll need

- A Postgres database. Absurd stores its task state there.
- A Pydantic AI `Agent` - so an LLM provider key (we'll use `openai:gpt-5.2`).
- `pip install pydantic-ai-absurd`.

!!! tip "No Postgres handy?"
    Start one in Docker, at the exact DSN used below:

    ```bash
    docker run -d --name absurd-postgres \
        -e POSTGRES_DB=absurd \
        -e POSTGRES_PASSWORD=postgres \
        -p 5432:5432 \
        postgres:16-alpine
    ```

    That's it - `postgresql://postgres:postgres@localhost:5432/absurd` is ready. (If you've cloned the repo, `scripts/postgres` does the same thing for you.)

The first time you connect, Absurd needs its schema and a queue. You do this **once**, at setup time:

```python
from absurd_sdk import AsyncAbsurd

absurd = AsyncAbsurd("postgresql://postgres:postgres@localhost:5432/absurd", queue_name="agents")
await absurd.create_queue()  # creates the 'agents' queue if it doesn't exist
```

!!! note
    Installing the Absurd schema itself is a one-time migration step that ships with `absurd-sdk`. Think of it like running your database migrations before the app starts - you do it on deploy, not on every run.

## Step 1: Wrap your agent

Start with an ordinary Pydantic AI agent. Then wrap it.

```python hl_lines="4 5"
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent

inner = Agent("openai:gpt-5.2", name="analyst")
agent = AbsurdAgent(inner, absurd)
```

That's the only change to your agent. `AbsurdAgent` keeps everything about `inner` - its model, its tools, its output type - but swaps the model (and any MCP tools) for versions that checkpoint each call.

!!! warning "The agent needs a name"
    The `name` isn't decoration - Absurd uses it as the prefix for every checkpoint step, so two agents with durable steps need two distinct names. Here it comes from the inner `Agent(..., name="analyst")`, and `AbsurdAgent` reuses it. If your inner agent has no name, pass one to `AbsurdAgent` directly: `AbsurdAgent(inner, absurd, name="analyst")`. Either way, if there's no name at all you'll get a clear error.

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

- **You write the task.** This is the same shape as Pydantic AI's Temporal integration - you control the workflow, and the agent is one durable step within it. You can do other things in here too: branch, call the agent twice, log, whatever.
- `params` is whatever you pass when you spawn the task (more on that in a second). It's plain JSON.
- `ctx` is the Absurd task context. You usually don't touch it directly - the wrapped agent uses it under the hood to record checkpoints.
- The return value is the task's result, stored in Postgres. Keep it JSON-serializable.

## Step 3: Spawn it

Now, from anywhere - a FastAPI endpoint, a cron job, a "Generate report" button - you ask for the task to run:

```python
handle = await absurd.spawn("analyse", {"prompt": "Analyse Q3 revenue"})
print(handle["task_id"])
```

Here's the important part: **`spawn` doesn't run the agent.** It writes a row to Postgres saying "someone please run `analyse` with these params" and returns immediately. Your web request finishes in milliseconds. The actual work happens elsewhere, later, durably.

!!! tip "This is why your API stays fast"
    The slow, expensive agent run never blocks the request that triggered it. You spawn and move on.

## Step 4: Run a worker

Something has to actually *do* the work. That's a worker - usually a separate process:

```python
# worker.py
async def main():
    absurd = AsyncAbsurd("postgresql://postgres:postgres@localhost:5432/absurd", queue_name="agents")

    inner = Agent("openai:gpt-5.2", name="analyst")
    agent = AbsurdAgent(inner, absurd)

    @absurd.register_task(name="analyse")
    async def analyse(params, ctx):
        result = await agent.run(params["prompt"])
        return {"output": result.output}

    await absurd.start_worker()  # claims tasks and runs them, forever
```

The worker polls Postgres, claims your spawned task, runs the `analyse` function, and stores the result. `start_worker()` runs until you stop it.

!!! warning "Register your tasks in the worker"
    The worker can only run tasks it knows about. The `@register_task` decorator must run **in the worker process** before `start_worker()`. The process that *spawns* doesn't need it - it only writes a task name and params to the database.

That's the full loop. Spawn from one place, run from another, talk only through Postgres.

## Step 5: Get the result

The task stored its return value. You can fetch it by `task_id`:

```python
result = await absurd.fetch_task_result(handle["task_id"])
if result is not None and result.state == "completed":
    print(result.result["output"])
```

In a real app you'd typically have the worker write the result somewhere your users can see - a row in your own table, a webhook, a websocket push. `fetch_task_result` is the simple polling version.

## Step 6: The payoff - crash and resume

Here's the whole reason we did any of this.

Imagine the `analyse` agent makes two model calls (say, it uses a tool in between). The worker finishes the **first** model call, records its checkpoint... and then the machine dies.

What happens?

1. Absurd notices the task didn't finish and makes it claimable again.
2. A new worker claims it and runs `analyse` from the top.
3. `agent.run()` reaches the first model call - but that one is already checkpointed. Instead of calling the LLM again, it returns the **cached** response instantly.
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
- [x] `start_worker` in a worker process
- [x] Let crashes resume instead of restart

Now that you've *seen* it work, the next page explains exactly **[how durability works](durability.md)** under the hood - what counts as a checkpoint, what doesn't, and the one surprise to watch out for.
