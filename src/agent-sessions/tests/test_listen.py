from __future__ import annotations

import anyio
import pytest

from agent_sessions import Session, SessionEvent

from .conftest import AsyncPool

pytestmark = pytest.mark.anyio


async def test_listen_receives_events_appended_concurrently(pool: AsyncPool, db_dsn: str) -> None:
    session = await Session.create(pool)

    received: list[SessionEvent] = []
    ready = anyio.Event()

    async def listener() -> None:
        async for event in session.listen(conninfo=db_dsn, timeout=3.0, stop_after=2, ready=ready):
            received.append(event)

    async def writer() -> None:
        await ready.wait()
        await session.append(kind='user_message', actor='user', payload={'i': 1})
        await session.append(kind='user_message', actor='user', payload={'i': 2})

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            tg.start_soon(listener)
            tg.start_soon(writer)

    assert [e.payload['i'] for e in received] == [1, 2]
    assert [e.sequence for e in received] == [1, 2]


async def test_listen_without_ready_event_and_timeout(pool: AsyncPool, db_dsn: str) -> None:
    """Call listen() with no `ready=` and a short timeout; it exits cleanly with no events."""
    session = await Session.create(pool)
    received: list[SessionEvent] = []
    async for event in session.listen(conninfo=db_dsn, timeout=0.2):
        received.append(event)  # pragma: no cover - timeout fires before any notify
    assert received == []
