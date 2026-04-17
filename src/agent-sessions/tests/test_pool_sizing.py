"""Regression test: concurrent brains on different sessions must not starve the pool.

The advisory lock that enforces `concurrency='queue'` must be held on a connection
that ISN'T also counted against the pool's quota for the brain body - otherwise
N concurrent sessions with pool.max_size=N saturate the pool and session.append()
inside the brain deadlocks waiting for a connection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import anyio
import pytest
from absurd_sdk import AsyncAbsurd
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from agent_sessions import BrainContext, Session, Workflow, apply_migrations

from .conftest import AsyncPool

pytestmark = pytest.mark.anyio


ABSURD_SQL = Path(
    '/Users/marcelotryle/dev/pydantic/agent-workflow/src/pydantic-ai-absurd/tests/fixtures/absurd.sql'
).read_text()


@pytest.fixture
async def small_pool(db_dsn: str) -> AsyncIterator[AsyncPool]:
    """A pool with only 2 connections - barely enough for basic operations, nowhere near enough
    if each brain pins a connection for its entire body."""
    async with AsyncConnectionPool(db_dsn, min_size=2, max_size=2, open=False) as pool:
        await pool.open(wait=True)
        await apply_migrations(pool)
        yield pool


@pytest.fixture
async def small_pool_absurd(db_dsn: str) -> AsyncIterator[AsyncAbsurd]:
    queue = f'pool_{uuid4().hex[:8]}'
    async with await AsyncConnection.connect(db_dsn, autocommit=True) as conn:
        client = AsyncAbsurd(conn, queue_name=queue)
        await client.create_queue()
        try:
            yield client
        finally:
            await client.drop_queue()


async def test_concurrent_brains_on_different_sessions_dont_starve_the_pool(
    small_pool: AsyncPool, small_pool_absurd: AsyncAbsurd
) -> None:
    """Spawn more concurrent brains than the pool has connections. All should complete.

    Previously this deadlocked because each brain pinned one of the 2 pool slots on its
    advisory-lock connection, and then session.append() on a third brain couldn't get a
    conn from the pool.
    """
    workflow = Workflow(absurd=small_pool_absurd, pool=small_pool)

    @workflow.brain('worker')
    async def worker(ctx: BrainContext[None]) -> None:
        # Simulate a brain that does actual work - multiple appends. Each needs a
        # pool connection for the duration of the append().
        await ctx.post('start')
        await anyio.sleep(0.05)
        await ctx.post('mid')
        await ctx.post('end')

    sessions = [await Session.create(small_pool) for _ in range(4)]
    for s in sessions:
        await workflow.wake(s, 'worker')

    # Drain with a time budget. If the pool starves, this will hang.
    with anyio.fail_after(15):
        # Absurd's work_batch processes tasks sequentially; to actually exercise the
        # concurrent-sessions case we need concurrent workers. Drive four parallel
        # work_batch loops - each claims and runs a task in parallel.
        async with anyio.create_task_group() as tg:

            async def drain() -> None:
                for _ in range(5):
                    await small_pool_absurd.work_batch(batch_size=1)

            for _ in range(4):
                tg.start_soon(drain)

    # All four sessions should have completed their brain runs.
    for s in sessions:
        events = await s.events()
        kinds = [e.kind for e in events]
        assert 'brain_started' in kinds, f'session {s.id} never started'
        assert 'brain_finished' in kinds, f'session {s.id} never finished'
