---
icon: lucide/life-buoy
---

# Troubleshooting

Most of the errors you'll hit early aren't bugs in your code. They're setup steps that haven't happened yet: a database that doesn't exist, a schema that wasn't installed, a worker that was never told about your task.

This page collects the ones you're most likely to see, with the exact message and the fix. If you're staring at a traceback, search this page for the line in bold.

## Setup, once

Before any of the examples work, two one-time things have to be true about your database:

1. **The Absurd schema is installed.** Absurd stores everything (tasks, runs, checkpoints) in an `absurd` schema made of SQL functions and tables. `absurd-sdk` does *not* create this for you, and it does not ship a CLI to do it. It's a plain SQL install you run once, the way you'd run any database migration. The SQL lives in the [Absurd repository](https://github.com/earendil-works/absurd); this project also vendors a copy at `tests/fixtures/absurd.sql`, so during development you can load it with:

    ```bash
    psql "postgresql://localhost/absurd" -f tests/fixtures/absurd.sql
    ```

2. **The queue exists.** Once the schema is in place, `await absurd.create_queue()` creates the queue your tasks live on. The example scripts call it inside `main()`.

If you skip either, you'll see one of the errors below.

## Errors you'll actually see

### `connection failed: ... role "..." does not exist`

You're connecting to a *different* Postgres than you think. This usually means something else (a local Homebrew Postgres, say) is already listening on port 5432, and your DSN's username doesn't exist on it.

Check what's on the port:

```bash
lsof -nP -iTCP:5432 -sTCP:LISTEN
```

If a local Postgres is squatting there, either point your DSN at it (with a role that exists, often your own username) or stop it so the database you intended is the one answering. The DSN is the source of truth: `postgresql://USER@HOST:PORT/DBNAME`.

### `database "..." does not exist`

The role connected fine, but the database in your DSN was never created. Create it:

```bash
createdb absurd
```

Then make sure your DSN's database name matches (`.../absurd`).

### `schema "absurd" does not exist`

You connected to the right database, but the Absurd schema isn't installed in it. This is step 1 above: load the Absurd SQL into that database once.

```bash
psql "postgresql://localhost/absurd" -f tests/fixtures/absurd.sql
```

A symptom of the same problem is an error mentioning `absurd.spawn_task` or `absurd.create_queue` not existing: those are the schema's functions, and they're missing because the schema is.

### `AbsurdAgent.run() must be called from inside an Absurd task handler`

You called `await agent.run(...)` directly, not inside a task. The durability comes from running *inside* a task, where Absurd can checkpoint each step, so that's the only place it's allowed.

Wrap it:

```python
@absurd.register_task(name="analyse")
async def analyse(params, ctx):
    result = await agent.run(params["prompt"])  # inside a task: works
    return {"output": result.output}
```

If you just want a plain, non-durable run for a quick test, use a normal Pydantic AI `Agent` instead of `AbsurdAgent`.

### `Unknown task` (the worker fails the task)

A worker claimed a task whose name it doesn't recognize. The fix is almost always: **register the task in the process that runs the worker.** `@register_task` has to execute in the worker process before `work_batch` or `start_worker`. The process that only *spawns* doesn't register anything, it just writes a name and params to the database, but the worker must know that name.

### `An agent needs a unique name to be used with Absurd`

The wrapped agent has no name, and Absurd uses the name as the prefix for every checkpoint. Give the inner agent a name (`Agent("openai:gpt-5.2", name="analyst")`) or pass one to `AbsurdAgent(inner, absurd, name="analyst")`.

### `Non-Absurd model cannot be overridden at run time`

You passed `model=...` to `agent.run(...)` (or `agent.override(...)`). The wrapped model *is* the durable model, so it's fixed when you build the `AbsurdAgent`. Set it at construction and don't override it per run.

## "It runs, but the script never exits"

Not an error, just a surprise. `await absurd.start_worker()` is meant to run forever: it polls Postgres and keeps claiming tasks, which is exactly what you want for a long-lived worker process.

For a script you want to *finish* (a scratch file, a test), use `work_batch` instead. It drains the tasks that are waiting and returns:

```python
await absurd.work_batch(batch_size=1)
```

See the [tutorial](tutorial.md) for where each one fits.

## Still stuck?

If it's the durability behavior that's confusing rather than the setup, [How durability works](durability.md) walks through exactly what's checkpointed and what happens on a crash. For anything that looks like an Absurd-level problem (the schema, queues, task states), the [Absurd project](https://github.com/earendil-works/absurd) is the source of truth.
