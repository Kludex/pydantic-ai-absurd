"""Runnable version of the README snippet.

Spins up a Postgres testcontainer, installs the Absurd + agent-sessions schemas,
builds the exact Workflow + Starlette app shown in the README, starts a worker
in the background, POSTs a user message via Starlette's TestClient, and prints
the resulting session events.

Run with:

    OPENAI_API_KEY=... uv run python examples/readme_snippet.py
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID, uuid4

import anyio
import psycopg
from absurd_sdk import AsyncAbsurd
from agent_sessions import Session, Workflow, apply_migrations
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from testcontainers.postgres import PostgresContainer

ABSURD_SQL = (
    Path(__file__).resolve().parent.parent / 'src' / 'pydantic-ai-absurd' / 'tests' / 'fixtures' / 'absurd.sql'
).read_text()


class PlanResult(BaseModel):
    reply: str
    needs_analyst: bool = False


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

            queue = f'readme_{uuid4().hex[:8]}'
            async with await AsyncConnection.connect(dsn, autocommit=True) as absurd_conn:
                absurd = AsyncAbsurd(absurd_conn, queue_name=queue)
                await absurd.create_queue()

                # ----- everything below matches the README snippet -----
                workflow = Workflow(absurd=absurd, pool=pool)

                planner = AbsurdAgent(
                    Agent('openai:gpt-5.2', name='planner', output_type=PlanResult),
                    absurd,
                )
                analyst = AbsurdAgent(Agent('openai:gpt-5.2', name='analyst'), absurd)

                @workflow.brain('planner')
                async def planner_brain(ctx):  # type: ignore[no-untyped-def]
                    result = await ctx.agent_run(planner, 'what should we do next?')
                    await ctx.post(result.output.reply)
                    if result.output.needs_analyst:
                        await ctx.wake('analyst')

                @workflow.brain('analyst')
                async def analyst_brain(ctx):  # type: ignore[no-untyped-def]
                    await ctx.post_status('analyzing...')
                    result = await ctx.agent_run(analyst, 'run a deep analysis')
                    await ctx.post(result.output)

                async def post_message(request: Request) -> JSONResponse:
                    body = await request.json()
                    session = await Session.load(pool, UUID(request.path_params['session_id']))
                    await session.append(
                        kind='user_message', actor='user', payload={'content': body['content']}
                    )
                    await workflow.wake(session, 'planner')
                    return JSONResponse({'ok': True})

                async def get_events(request: Request) -> JSONResponse:
                    session = await Session.load(pool, UUID(request.path_params['session_id']))
                    events = await session.events(after=int(request.query_params.get('after', 0)))
                    return JSONResponse([e.model_dump(mode='json') for e in events])

                app = Starlette(
                    routes=[
                        Route('/sessions/{session_id}/messages', post_message, methods=['POST']),
                        Route('/sessions/{session_id}/events', get_events, methods=['GET']),
                    ]
                )
                # ----- end README snippet -----

                # Driver: create a session, hit the HTTP endpoint, drain the worker, read events.
                session = await Session.create(pool)
                with TestClient(app) as client:
                    r = client.post(f'/sessions/{session.id}/messages', json={'content': 'hi'})
                    assert r.status_code == 200, r.text

                # Drain Absurd (this stands in for `await workflow.run()` in a worker process).
                for _ in range(5):
                    await absurd.work_batch(batch_size=4)

                with TestClient(app) as client:
                    r = client.get(f'/sessions/{session.id}/events')
                    events = r.json()

                print(f'{len(events)} events on session {session.id}:')
                for e in events:
                    print(f'  [{e["sequence"]}] {e["kind"]:<18} actor={e["actor"]:<16} payload={e["payload"]}')


if __name__ == '__main__':
    anyio.run(main)
