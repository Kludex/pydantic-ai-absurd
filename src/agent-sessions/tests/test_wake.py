from __future__ import annotations

from uuid import uuid4

import pytest
from absurd_sdk import AsyncAbsurd

from agent_sessions import BrainContext, Session, brain, create_worker, wake
from agent_sessions.wake import brain_task_name

from .conftest import AsyncPool

pytestmark = pytest.mark.anyio


async def test_deterministic_dedup_key_collapses_duplicate_wakes(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('x')
    async def x_brain(ctx: BrainContext[None]) -> None:
        await ctx.post('once')  # pragma: no cover - not drained

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)

    first = await wake(absurd, session, 'x', input={'k': 'v'})
    second = await wake(absurd, session, 'x', input={'k': 'v'})

    assert first.task_id == second.task_id
    assert second.deduplicated is True
    assert first.deduplicated is False


async def test_explicit_dedup_key_collapses(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('y')
    async def y_brain(ctx: BrainContext[None]) -> None:
        await ctx.post('p')  # pragma: no cover - not drained

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)

    first = await wake(absurd, session, 'y', dedup_key='custom', input={'k': 1})
    second = await wake(absurd, session, 'y', dedup_key='custom', input={'k': 2})
    assert first.task_id == second.task_id


async def test_different_inputs_spawn_separate_tasks(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('z')
    async def z_brain(ctx: BrainContext[None]) -> None:
        pass  # pragma: no cover

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)

    first = await wake(absurd, session, 'z', input={'q': 1})
    second = await wake(absurd, session, 'z', input={'q': 2})
    assert first.task_id != second.task_id


async def test_wake_requires_session_or_pool(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    with pytest.raises(ValueError, match='Session or an explicit pool'):
        await wake(absurd, uuid4(), 'anything')


async def test_wake_with_explicit_pool_and_session_id(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('w')
    async def w_brain(ctx: BrainContext[None]) -> None:
        pass  # pragma: no cover

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    handle = await wake(absurd, session.id, 'w', pool=pool)
    assert handle.task_id
    assert not handle.deduplicated


async def test_wake_parallel_runs_immediately(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('p')
    async def p_brain(ctx: BrainContext[None]) -> None:
        await ctx.post('done')  # pragma: no cover - not drained

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    handle = await wake(absurd, session, 'p', concurrency='parallel')
    assert not handle.deduplicated


async def test_supersede_cancels_existing_tasks(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('s')
    async def s_brain(ctx: BrainContext[None]) -> None:
        pass  # pragma: no cover

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    first = await wake(absurd, session, 's', input={'v': 1})
    second = await wake(absurd, session, 's', input={'v': 2}, concurrency='supersede')
    assert second.task_id != first.task_id


def test_brain_task_name() -> None:
    assert brain_task_name('foo') == 'brain.foo'
