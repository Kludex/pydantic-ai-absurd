"""Runs the Vercel AI + Starlette example on a real port so you can hit it from a browser or curl.

    OPENAI_API_KEY=... uv run python examples/vercel_starlette_server.py

Prints the session_id on startup. POST to /sessions/{id}/chat with the useChat
payload shape. GET / serves a tiny HTML page that does the streaming for you.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import psycopg
import uvicorn
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
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route
from testcontainers.postgres import PostgresContainer

ABSURD_SQL = (
    Path(__file__).resolve().parent.parent / 'src' / 'pydantic-ai-absurd' / 'tests' / 'fixtures' / 'absurd.sql'
).read_text()


INDEX_HTML = """<!doctype html>
<meta charset="utf-8">
<title>agent-workflow Vercel demo</title>
<style>
  body { font: 14px/1.5 -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
  #log { white-space: pre-wrap; border: 1px solid #ccc; padding: 12px; min-height: 160px; border-radius: 6px; }
  textarea { width: 100%; box-sizing: border-box; }
  button { padding: 8px 16px; }
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
</style>
<h1>agent-workflow Vercel demo</h1>
<p>Session id: <code id="sid">__SESSION_ID__</code></p>
<textarea id="msg" rows="3">hello in three words</textarea>
<button id="send">Send</button>
<div id="log"></div>
<script>
const sid = document.getElementById('sid').textContent;
const log = document.getElementById('log');
document.getElementById('send').onclick = async () => {
  log.textContent = '';
  const res = await fetch(`/sessions/${sid}/chat`, {
    method: 'POST',
    headers: {'content-type': 'application/json', 'accept': 'text/event-stream'},
    body: JSON.stringify({
      id: crypto.randomUUID(),
      trigger: 'submit-message',
      messages: [{
        id: crypto.randomUUID(),
        role: 'user',
        parts: [{type: 'text', text: document.getElementById('msg').value}],
      }],
    }),
  });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    for (;;) {
      const i = buf.indexOf('\\n');
      if (i < 0) break;
      const line = buf.slice(0, i);
      buf = buf.slice(i + 1);
      if (!line.startsWith('data: ')) continue;
      const body = line.slice(6);
      if (body === '[DONE]') continue;
      try {
        const chunk = JSON.parse(body);
        if (chunk.type === 'text-delta') log.textContent += chunk.delta;
      } catch (_) {}
    }
  }
};
</script>
"""


def build_app(pool: AsyncConnectionPool, workflow: Workflow, session_id: UUID) -> Starlette:
    chat_agent = Agent('openai:gpt-5.2', name='chat')

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(INDEX_HTML.replace('__SESSION_ID__', str(session_id)))

    async def chat_endpoint(request: Request) -> Response:
        session = await Session.load(pool, UUID(request.path_params['session_id']))
        history = await session.messages()

        async def persist_result(result: AgentRunResult[object]) -> None:
            for kwargs in messages_to_events(list(result.new_messages()), actor='chat'):
                await session.append(**kwargs)

        return await VercelAIAdapter.dispatch_request(
            request, agent=chat_agent, message_history=history, on_complete=persist_result,
        )

    async def events_endpoint(request: Request) -> JSONResponse:
        session = await Session.load(pool, UUID(request.path_params['session_id']))
        events = await session.events(after=int(request.query_params.get('after', 0)))
        return JSONResponse([json.loads(e.model_dump_json()) for e in events])

    return Starlette(
        routes=[
            Route('/', index, methods=['GET']),
            Route('/sessions/{session_id}/chat', chat_endpoint, methods=['POST']),
            Route('/sessions/{session_id}/events', events_endpoint, methods=['GET']),
        ]
    )


class AppState:
    app: Starlette | None = None


state = AppState()


async def _startup() -> None:
    if 'DOCKER_HOST' not in os.environ:
        home_sock = Path.home() / '.docker' / 'run' / 'docker.sock'
        if home_sock.exists():
            os.environ['DOCKER_HOST'] = f'unix://{home_sock}'

    container = PostgresContainer('postgres:16-alpine')
    container.start()
    dsn = container.get_connection_url().replace('postgresql+psycopg2://', 'postgresql://')
    with psycopg.connect(dsn, autocommit=True) as cx:
        cx.execute(ABSURD_SQL)

    pool: AsyncConnectionPool = AsyncConnectionPool(dsn, min_size=1, max_size=4, open=False)
    await pool.open(wait=True)
    await apply_migrations(pool)

    queue = f'vercel_{uuid4().hex[:8]}'
    absurd_conn = await AsyncConnection.connect(dsn, autocommit=True)
    absurd = AsyncAbsurd(absurd_conn, queue_name=queue)
    await absurd.create_queue()
    workflow = Workflow(absurd=absurd, pool=pool)

    session = await Session.create(pool)
    state.app = build_app(pool, workflow, session.id)
    print(f'\n  Open:   http://127.0.0.1:8000/')
    print(f'  Session id: {session.id}')
    print(f'  Direct:  curl -N -X POST http://127.0.0.1:8000/sessions/{session.id}/chat ...\n')


async def entry(scope, receive, send):  # type: ignore[no-untyped-def]
    if state.app is None:
        await _startup()
    assert state.app is not None
    await state.app(scope, receive, send)


if __name__ == '__main__':
    uvicorn.run(entry, host='127.0.0.1', port=8000, log_level='info')
