from __future__ import annotations

from uuid import uuid4

import pytest

from agent_sessions import Session, Visibility

from .conftest import AsyncPool

pytestmark = pytest.mark.anyio


async def test_create_and_load(pool: AsyncPool) -> None:
    session = await Session.create(pool, metadata={'owner': 'alice'})
    loaded = await Session.load(pool, session.id)
    assert loaded.id == session.id


async def test_load_missing_raises(pool: AsyncPool) -> None:
    with pytest.raises(LookupError):
        await Session.load(pool, uuid4())


async def test_append_assigns_sequential_numbers(pool: AsyncPool) -> None:
    session = await Session.create(pool)
    first = await session.append(kind='user_message', actor='user', payload={'content': 'hi'})
    second = await session.append(kind='assistant_message', actor='brain:x', payload={'content': 'hello'})
    assert (first.sequence, second.sequence) == (1, 2)


async def test_events_with_after_cursor(pool: AsyncPool) -> None:
    session = await Session.create(pool)
    await session.append(kind='user_message', actor='user', payload={'i': 1})
    second = await session.append(kind='user_message', actor='user', payload={'i': 2})
    await session.append(kind='user_message', actor='user', payload={'i': 3})
    tail = await session.events(after=second.sequence)
    assert [e.payload['i'] for e in tail] == [3]


async def test_events_respects_visibility_filter(pool: AsyncPool) -> None:
    session = await Session.create(pool)
    await session.append(kind='user_message', actor='user', payload={'i': 1}, visibility=Visibility.public)
    await session.append(kind='tool_call', actor='brain:x', payload={'i': 2}, visibility=Visibility.internal)
    public_only = await session.events(visibility=Visibility.public)
    assert [e.payload['i'] for e in public_only] == [1]


async def test_events_honours_limit(pool: AsyncPool) -> None:
    session = await Session.create(pool)
    for i in range(5):
        await session.append(kind='user_message', actor='user', payload={'i': i})
    limited = await session.events(limit=2)
    assert len(limited) == 2


async def test_snapshot_round_trip(pool: AsyncPool) -> None:
    session = await Session.create(pool)
    await session.append(kind='user_message', actor='user', payload={'i': 1})
    snap = await session.create_snapshot(up_to_sequence=1, summary_payload={'summary': 'one event'})
    latest = await session.latest_snapshot()
    assert latest is not None
    assert latest.up_to_sequence == snap.up_to_sequence
    assert latest.summary_payload == {'summary': 'one event'}


async def test_latest_snapshot_none_when_absent(pool: AsyncPool) -> None:
    session = await Session.create(pool)
    assert await session.latest_snapshot() is None


async def test_session_id_property_and_pool(pool: AsyncPool) -> None:
    session = await Session.create(pool)
    assert session.pool is pool
    assert session.id == session.id  # stable
