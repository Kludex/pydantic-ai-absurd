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
from uuid import UUID

from absurd_sdk import AsyncAbsurd
from agent_sessions import Session, Workflow
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

class PlanResult(BaseModel):
    reply: str
    needs_analyst: bool = False

workflow = Workflow(absurd=absurd, pool=pool)

planner = AbsurdAgent(
    Agent('openai:gpt-5.2', name='planner', output_type=PlanResult),
    absurd,
)
analyst = AbsurdAgent(Agent('openai:gpt-5.2', name='analyst'), absurd)

@workflow.brain('planner')
async def planner_brain(ctx):
    result = await ctx.agent_run(planner, 'what should we do next?')
    await ctx.post(result.output.reply)
    if result.output.needs_analyst:
        await ctx.wake('analyst')

@workflow.brain('analyst')
async def analyst_brain(ctx):
    await ctx.post_status('analyzing...')
    result = await ctx.agent_run(analyst, 'run a deep analysis')
    await ctx.post(result.output)

# HTTP side (Starlette)
async def post_message(request: Request) -> JSONResponse:
    body = await request.json()
    session = await Session.load(pool, UUID(request.path_params['session_id']))
    await session.append(kind='user_message', actor='user', payload={'content': body['content']})
    await workflow.wake(session, 'planner')
    return JSONResponse({'ok': True})

async def get_events(request: Request) -> JSONResponse:
    session = await Session.load(pool, UUID(request.path_params['session_id']))
    events = await session.events(after=int(request.query_params.get('after', 0)))
    return JSONResponse([e.model_dump(mode='json') for e in events])

app = Starlette(routes=[
    Route('/sessions/{session_id}/messages', post_message, methods=['POST']),
    Route('/sessions/{session_id}/events', get_events, methods=['GET']),
])

# Worker side (separate process - brains are registered with Absurd the moment
# they're decorated, so just start the loop)
await workflow.run()
```

The worker survives restarts mid-run: when it comes back, Absurd replays from the last checkpoint and the brain continues where it left off. The Starlette process stays stateless - it only appends to the session and fires `wake()`.

## Stream an agent to a Vercel AI SDK client

For the user's HTTP turn - where the browser expects streaming SSE compatible with [`useChat`](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot) - use Pydantic AI's built-in `VercelAIAdapter` on top of a Starlette route. Message history comes from the session; on completion we persist the new messages via the agent-sessions adapter so a follow-up brain can operate durably.

```python
from agent_sessions import Session
from agent_sessions._pydantic_ai import messages_to_events
from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from starlette.routing import Route

chat = Agent('openai:gpt-5.2', name='chat')

async def chat_endpoint(request):
    session = await Session.load(pool, UUID(request.path_params['session_id']))
    history = await session.messages()

    async def persist(result):
        for kwargs in messages_to_events(list(result.new_messages()), actor='chat'):
            await session.append(**kwargs)

    return await VercelAIAdapter.dispatch_request(
        request, agent=chat, message_history=history, on_complete=persist,
    )

app = Starlette(routes=[Route('/sessions/{session_id}/chat', chat_endpoint, methods=['POST'])])
```

A runnable version is in `examples/vercel_starlette.py` - it spins up Postgres, streams real GPT-5.2 deltas, and verifies the session picks up the resulting messages.

## Develop

```bash
scripts/install   # uv sync --all-packages
scripts/check     # ruff format --check + ruff check + mypy strict
scripts/test      # pytest + 100% coverage gate (both packages)
```

Tests run against a real Postgres via `testcontainers`. Docker must be up.

The example scripts under `examples/` can be smoke-tested end-to-end against real OpenAI (kept out of CI - local use only):

```bash
OPENAI_API_KEY=... uv run pytest examples/tests/
```

The tests auto-skip when `OPENAI_API_KEY` isn't set.
