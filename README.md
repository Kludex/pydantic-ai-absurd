# Pydantic AI Absurd

<p align="center"><em>Durable execution for Pydantic AI agents, on Postgres alone.</em></p>

---

Agents run for a while, a model call, a tool call, another model call. When the worker dies in the middle of that, the run is usually lost: you restart from zero and pay for every token again.

**Pydantic AI Absurd** fixes that. Call `agent.run()` inside a durable task and every model and MCP call is checkpointed into Postgres. If the worker crashes, a new one resumes from the last completed step, no restart, no re-spent tokens. Same idea as Pydantic AI's Temporal integration, but with no Temporal, no Redis, no broker: just the Postgres you already have.

## Installation

```bash
pip install pydantic-ai-absurd
```

## Example

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
    await absurd.spawn("analyse", {"prompt": "Analyse Q3 revenue"})
    await absurd.work_batch(batch_size=1)


if __name__ == "__main__":
    asyncio.run(main())
```

You author a task, call the agent inside it, and run it durably. That's the whole idea.

## Documentation

Read the docs at **[kludex.github.io/pydantic-ai-absurd](https://kludex.github.io/pydantic-ai-absurd/)**, a step-by-step tutorial, how durability actually works, tools and MCP servers, and a production guide.
