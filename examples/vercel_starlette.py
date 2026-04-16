"""Runnable Vercel AI + Starlette + agent-sessions example.

What this shows: a Starlette endpoint that speaks the Vercel AI SDK's `useChat`
protocol, driven by a Pydantic AI `Agent`, threading message history through an
`agent-sessions` `Session`. On completion we append the resulting messages to
the session so a durable follow-up brain could be woken for any post-chat work
(e.g. summarisation, analysis) without re-running the chat model.

Run end-to-end against a real Postgres (testcontainer) + OpenAI:

    OPENAI_API_KEY=... uv run python examples/vercel_starlette.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import anyio
import psycopg
from absurd_sdk import AsyncAbsurd
from agent_sessions import Session, Workflow, apply_migrations
from agent_sessions._pydantic_ai import messages_to_events
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool
from pydantic_ai import Agent
from pydantic_ai.run import AgentRunResult
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient
from testcontainers.postgres import PostgresContainer

ABSURD_SQL = (
    Path(__file__).resolve().parent.parent / 'src' / 'pydantic-ai-absurd' / 'tests' / 'fixtures' / 'absurd.sql'
).read_text()


def build_app(pool: AsyncConnectionPool, workflow: Workflow) -> Starlette:
    chat_agent = Agent('openai:gpt-5.2', name='chat')

    async def chat_endpoint(request: Request) -> Response:
        session_id = UUID(request.path_params['session_id'])
        session = await Session.load(pool, session_id)
        history = await session.messages()

        async def persist_result(result: AgentRunResult[object]) -> None:
            for kwargs in messages_to_events(list(result.new_messages()), actor='chat'):
                await session.append(**kwargs)

        # VercelAIAdapter parses the useChat POST body, streams the agent as SSE,
        # and calls `on_complete` with the final AgentRunResult.
        return await VercelAIAdapter.dispatch_request(
            request,
            agent=chat_agent,
            message_history=history,
            on_complete=persist_result,
        )

    return Starlette(routes=[Route('/sessions/{session_id}/chat', chat_endpoint, methods=['POST'])])


async def main() -> None:
    if 'DOCKER_HOST' not in os.environ:
        home_sock = Path.home() / '.docker' / 'run' / 'docker.sock'
        if home_sock.exists():
            os.environ['DOCKER_HOST'] = f'unix://{home_sock}'

    with PostgresContainer('postgres:16-alpine') as container:
        dsn = container.get_connection_url().replace('postgresql+psycopg2://', 'postgresql://')
        with psycopg.connect(dsn, autocommit=True) as cx:
            cx.execute(ABSURD_SQL)

        async with AsyncConnectionPool(dsn, min_size=1, max_size=4, open=False) as pool:
            await pool.open(wait=True)
            await apply_migrations(pool)

            queue = f'vercel_{uuid4().hex[:8]}'
            async with await AsyncConnection.connect(dsn, autocommit=True) as absurd_conn:
                absurd = AsyncAbsurd(absurd_conn, queue_name=queue)
                await absurd.create_queue()

                workflow = Workflow(absurd=absurd, pool=pool)
                app = build_app(pool, workflow)

                # Create a session, POST a useChat-style request, iterate the SSE response.
                session = await Session.create(pool)
                payload = {
                    'id': 'user-msg-1',
                    'messages': [
                        {
                            'id': 'user-msg-1',
                            'role': 'user',
                            'parts': [{'type': 'text', 'text': 'hello in three words'}],
                        }
                    ],
                    'trigger': 'submit-message',
                }

                with TestClient(app) as client:
                    with client.stream(
                        'POST',
                        f'/sessions/{session.id}/chat',
                        json=payload,
                        headers={'accept': 'text/event-stream'},
                    ) as response:
                        assert response.status_code == 200, response.read()
                        chunk_count = 0
                        for raw in response.iter_lines():
                            if not raw.startswith('data: '):
                                continue
                            body = raw[len('data: ') :]
                            if body == '[DONE]':
                                continue
                            chunk = json.loads(body)
                            if chunk.get('type') == 'text-delta':
                                print(f'delta: {chunk["delta"]!r}')
                            chunk_count += 1
                        print(f'\ntotal SSE chunks: {chunk_count}')

                events = await session.events()
                print(f'\n{len(events)} events now on session {session.id}:')
                for e in events:
                    print(f'  [{e.sequence}] {e.kind:<18} actor={e.actor}')


if __name__ == '__main__':
    anyio.run(main)
