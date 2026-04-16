# pydantic-ai-absurd

Durable execution adapter that runs Pydantic AI agents as [Absurd](https://github.com/earendil-works/absurd) tasks, so every model call and tool call is checkpointed into Postgres and resumes where it stopped after a crash.

## Install

```bash
uv add pydantic-ai-absurd
```

Requires Postgres (for Absurd) and a Pydantic AI `Agent`.

## Quickstart

```python
from absurd_sdk import AsyncAbsurd
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent

absurd = AsyncAbsurd('postgresql://localhost/absurd')
inner = Agent('anthropic:claude-sonnet-4-6', name='analyst')
agent = AbsurdAgent(inner, absurd, name='analyst')

# Call `agent.run(...)` from inside an Absurd task handler.
```

See the repository `README.md` for the full architecture story and `agent-sessions` for higher-level orchestration (brains, sessions, wake).
