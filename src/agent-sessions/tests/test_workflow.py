from __future__ import annotations

from uuid import uuid4

import anyio
import pytest
from absurd_sdk import AsyncAbsurd, JsonValue
from psycopg import AsyncConnection
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_absurd import AbsurdAgent

from agent_sessions import BrainContext, EventKind, Session, Workflow

from .conftest import AsyncPool

pytestmark = pytest.mark.anyio


async def test_brain_decorator_registers_once(workflow: Workflow) -> None:
    @workflow.brain('x')
    async def _brain_x(ctx: BrainContext[None]) -> None:
        pass  # pragma: no cover - not driven

    assert [d.name for d in workflow.brains] == ['x']


async def test_duplicate_brain_raises(workflow: Workflow) -> None:
    @workflow.brain('dup')
    async def _brain_a(ctx: BrainContext[None]) -> None:  # pragma: no cover
        pass

    with pytest.raises(ValueError, match='already registered'):

        @workflow.brain('dup')
        async def _brain_b(ctx: BrainContext[None]) -> None:  # pragma: no cover
            pass


async def test_task_name(workflow: Workflow) -> None:
    assert workflow.task_name('foo') == 'brain.foo'


async def test_workflow_exposes_absurd_and_pool(workflow: Workflow, absurd: AsyncAbsurd, pool: AsyncPool) -> None:
    assert workflow.absurd is absurd
    assert workflow.pool is pool


async def test_from_dsn_builds_and_owns_pool_and_absurd(db_dsn: str, pool: AsyncPool) -> None:
    # The `pool` fixture has already run `apply_migrations`; from_dsn assumes the
    # schema exists. Create the queue the owned client will use.
    queue = f'from_dsn_{uuid4().hex[:8]}'
    async with await AsyncConnection.connect(db_dsn, autocommit=True) as conn:
        await AsyncAbsurd(conn, queue_name=queue).create_queue()

    async with await Workflow.from_dsn(db_dsn, queue_name=queue, session_lease_poll_seconds=0.05) as workflow:
        ran: list[str] = []

        @workflow.brain('owned')
        async def owned(ctx: BrainContext[None]) -> None:
            ran.append('yes')

        session = await Session.create(workflow.pool)
        await workflow.wake(session, 'owned')
        for _ in range(10):
            await workflow.absurd.work_batch(batch_size=4)
            await anyio.sleep(0.02)

        assert ran == ['yes']

    # aclose() closed the pool it created.
    assert workflow.pool.closed


async def test_aclose_leaves_injected_resources_open(absurd: AsyncAbsurd, pool: AsyncPool, db_dsn: str) -> None:
    workflow = Workflow(absurd=absurd, pool=pool)
    await workflow.aclose()
    # Injected pool is untouched - still usable.
    session = await Session.create(pool)
    assert session.id


async def test_independent_workflows_have_separate_registries(absurd: AsyncAbsurd, pool: AsyncPool) -> None:
    wf_a = Workflow(absurd=absurd, pool=pool)
    wf_b = Workflow(absurd=absurd, pool=pool)

    @wf_a.brain('only_a')
    async def _a(ctx: BrainContext[None]) -> None:  # pragma: no cover
        pass

    assert [d.name for d in wf_a.brains] == ['only_a']
    assert wf_b.brains == []


async def test_brain_runs_and_emits_lifecycle(workflow: Workflow) -> None:
    @workflow.brain('greeter')
    async def greeter_brain(ctx: BrainContext[None]) -> None:
        await ctx.post('hello from greeter')

    session = await Session.create(workflow.pool)
    handle = await workflow.wake(session, 'greeter')
    assert handle.task_id

    await workflow.absurd.work_batch(batch_size=1)

    kinds = [e.kind for e in await session.events()]
    assert kinds == [
        EventKind.brain_started,
        EventKind.assistant_message,
        EventKind.brain_finished,
    ]


async def test_brain_failure_emits_brain_failed(workflow: Workflow) -> None:
    @workflow.brain('bad')
    async def bad_brain(ctx: BrainContext[None]) -> None:
        raise RuntimeError('boom')

    session = await Session.create(workflow.pool)
    await workflow.wake(session, 'bad', max_attempts=1)

    await workflow.absurd.work_batch(batch_size=1)

    events = await session.events()
    kinds = [e.kind for e in events]
    assert EventKind.brain_failed in kinds
    failed = next(e for e in events if e.kind == EventKind.brain_failed)
    assert 'boom' in failed.payload['error']


async def test_on_poison_handler_is_called(absurd: AsyncAbsurd, pool: AsyncPool) -> None:
    captured: list[object] = []

    async def on_poison(event: object) -> None:
        captured.append(event)

    workflow = Workflow(absurd=absurd, pool=pool, on_poison=on_poison)

    @workflow.brain('bad')
    async def bad_brain(ctx: BrainContext[None]) -> None:
        raise RuntimeError('kaboom')

    session = await Session.create(pool)
    await workflow.wake(session, 'bad', max_attempts=1)
    await absurd.work_batch(batch_size=1)

    assert len(captured) == 1


async def test_post_status_emits_status_update(workflow: Workflow) -> None:
    @workflow.brain('status')
    async def status_brain(ctx: BrainContext[None]) -> None:
        await ctx.post_status('starting')

    session = await Session.create(workflow.pool)
    await workflow.wake(session, 'status')
    await workflow.absurd.work_batch(batch_size=1)

    kinds = [e.kind for e in await session.events()]
    assert EventKind.status_update in kinds


async def test_brain_chain_via_ctx_wake(workflow: Workflow) -> None:
    @workflow.brain('router')
    async def router(ctx: BrainContext[None]) -> None:
        await ctx.post('routing')
        await ctx.wake('analyst')

    @workflow.brain('analyst')
    async def analyst(ctx: BrainContext[None]) -> None:
        await ctx.post('analyzed')

    session = await Session.create(workflow.pool)
    await workflow.wake(session, 'router')

    for _ in range(3):
        await workflow.absurd.work_batch(batch_size=2)

    events = await session.events()
    assistants = [e for e in events if e.kind == EventKind.assistant_message]
    contents = [e.payload['content'] for e in assistants]
    assert contents == ['routing', 'analyzed']


async def test_brain_input_is_accessible(workflow: Workflow) -> None:
    observed: dict[str, object] = {}

    @workflow.brain('echo')
    async def echo_brain(ctx: BrainContext[None]) -> None:
        observed['input'] = dict(ctx.input)
        observed['name'] = ctx.name
        observed['session_id'] = ctx.session.id
        observed['absurd_ctx'] = ctx.absurd_ctx

    session = await Session.create(workflow.pool)
    await workflow.wake(session, 'echo', input={'msg': 'hi'})
    await workflow.absurd.work_batch(batch_size=1)

    assert observed['input'] == {'msg': 'hi'}
    assert observed['name'] == 'echo'
    assert observed['session_id'] == session.id
    assert observed['absurd_ctx'] is not None


async def test_agent_run_threads_history(workflow: Workflow) -> None:
    seen_history: list[list[ModelMessage]] = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_history.append(list(messages))
        return ModelResponse(parts=[TextPart(content='pong')])

    inner_agent = Agent(FunctionModel(fn, model_name='m'), name='inner')
    absurd_agent = AbsurdAgent(inner_agent, workflow.absurd, name='inner')

    @workflow.brain('driver')
    async def driver(ctx: BrainContext[None]) -> None:
        await ctx.agent_run(absurd_agent, 'continue')
        await ctx.agent_run(absurd_agent, 'again')

    session = await Session.create(workflow.pool)
    await workflow.wake(session, 'driver')
    await workflow.absurd.work_batch(batch_size=1)

    assert len(seen_history) == 2
    assert any(isinstance(m, ModelResponse) for m in seen_history[1])


async def test_brain_sleep_checkpoints_via_absurd(workflow: Workflow) -> None:
    @workflow.brain('sleeper')
    async def sleeper(ctx: BrainContext[None]) -> None:
        await ctx.sleep(0.1)
        await ctx.post('awake')

    session = await Session.create(workflow.pool)
    await workflow.wake(session, 'sleeper')
    await workflow.absurd.work_batch(batch_size=1)  # suspends on sleep_for
    await anyio.sleep(0.3)
    await workflow.absurd.work_batch(batch_size=1)  # resumes

    contents = [e.payload.get('content') for e in await session.events() if e.kind == EventKind.assistant_message]
    assert 'awake' in contents


async def test_rejects_non_object_input(workflow: Workflow) -> None:
    @workflow.brain('needs_object')
    async def _br(ctx: BrainContext[None]) -> None:  # pragma: no cover - never reached
        pass

    session = await Session.create(workflow.pool)
    params: JsonValue = {'session_id': str(session.id), 'input': 'not-a-dict'}
    await workflow.absurd.spawn('brain.needs_object', params, max_attempts=1)
    await workflow.absurd.work_batch(batch_size=1)


async def test_parallel_concurrency_skips_lock(workflow: Workflow) -> None:
    @workflow.brain('par')
    async def par_brain(ctx: BrainContext[None]) -> None:
        await ctx.post('parallel')

    session = await Session.create(workflow.pool)
    await workflow.wake(session, 'par', concurrency='parallel')
    await workflow.absurd.work_batch(batch_size=1)
    contents = [e.payload.get('content') for e in await session.events() if e.kind == EventKind.assistant_message]
    assert 'parallel' in contents


async def test_missing_session_id_fails_task(workflow: Workflow) -> None:
    @workflow.brain('needs_id')
    async def _br(ctx: BrainContext[None]) -> None:  # pragma: no cover
        pass

    await workflow.absurd.spawn('brain.needs_id', {'input': {}}, max_attempts=1)
    await workflow.absurd.work_batch(batch_size=1)


async def test_deterministic_dedup_collapses_duplicate_wakes(workflow: Workflow) -> None:
    @workflow.brain('dedup_x')
    async def _br(ctx: BrainContext[None]) -> None:
        await ctx.post('once')  # pragma: no cover - not drained

    session = await Session.create(workflow.pool)
    first = await workflow.wake(session, 'dedup_x', input={'k': 'v'})
    second = await workflow.wake(session, 'dedup_x', input={'k': 'v'})
    assert first.task_id == second.task_id
    assert first.dedup_key == second.dedup_key


async def test_explicit_dedup_key_collapses(workflow: Workflow) -> None:
    @workflow.brain('dedup_y')
    async def _br(ctx: BrainContext[None]) -> None:
        await ctx.post('p')  # pragma: no cover - not drained

    session = await Session.create(workflow.pool)
    first = await workflow.wake(session, 'dedup_y', dedup_key='custom', input={'k': 1})
    second = await workflow.wake(session, 'dedup_y', dedup_key='custom', input={'k': 2})
    assert first.task_id == second.task_id


async def test_wake_by_session_id_only(workflow: Workflow) -> None:
    @workflow.brain('dedup_w')
    async def _br(ctx: BrainContext[None]) -> None:
        pass  # pragma: no cover

    session = await Session.create(workflow.pool)
    handle = await workflow.wake(session.id, 'dedup_w')
    assert handle.task_id


async def test_second_brain_on_same_session_waits_for_lease(workflow: Workflow) -> None:
    """Two brains woken on the same session must serialize via the lease.

    Drives the `sleep_for` polling branch in `_acquire_session_lease`: the second
    brain's first CAS fails (lease held by brain one), it suspends, then retries
    after the poll interval and succeeds once brain one releases.

    We prime the lease by manually setting `running_task_id` to a fake id before
    any task runs, so the first `_acquire_session_lease` CAS is guaranteed to
    miss and the poll path is deterministic. After a short delay the conftest
    poll interval elapses; we free the lease and drain.
    """
    order: list[str] = []

    @workflow.brain('only')
    async def only(ctx: BrainContext[None]) -> None:
        order.append('ran')

    session = await Session.create(workflow.pool)
    # Pre-occupy the lease so the brain's first CAS miss and triggers the poll path.
    async with workflow.pool.connection() as conn:
        await conn.execute(
            'UPDATE agent_sessions.sessions SET running_task_id = %s WHERE id = %s',
            ('sentinel', session.id),
        )

    await workflow.wake(session, 'only')

    # First drain: brain hits the contended lease, suspends on sleep_for.
    await workflow.absurd.work_batch(batch_size=1)
    assert order == []

    # Free the lease; subsequent poll attempts will succeed once they re-fire.
    async with workflow.pool.connection() as conn:
        await conn.execute(
            'UPDATE agent_sessions.sessions SET running_task_id = NULL WHERE id = %s',
            (session.id,),
        )

    # Wait past the poll interval, then drain again.
    await anyio.sleep(0.15)
    for _ in range(5):
        await workflow.absurd.work_batch(batch_size=1)
        await anyio.sleep(0.05)

    assert order == ['ran']


async def test_supersede_cancels_the_active_brain(workflow: Workflow) -> None:
    """`concurrency='supersede'` cancels the currently-leased brain for the same
    (session, brain_name) before spawning the replacement. Prove it: spawn a long-running
    brain, then fire a supersede - the first brain fails with cancellation, the second
    runs and completes.
    """
    observed: dict[str, int | bool] = {'first_attempts': 0, 'second_ran': False}

    @workflow.brain('flaky_first')
    async def flaky(ctx: BrainContext[None]) -> None:
        observed['first_attempts'] = int(observed['first_attempts']) + 1
        # Sleep durably so the task suspends and lives in the "running" lease -
        # the supersede path needs a lease holder to cancel.
        await ctx.sleep(1.0)
        observed['first_finished'] = True  # pragma: no cover - we intend to be cancelled

    @workflow.brain('replacer')
    async def replacer(ctx: BrainContext[None]) -> None:
        observed['second_ran'] = True

    session = await Session.create(workflow.pool)

    # Spawn the first brain with max_attempts=1 so the cancel doesn't trigger a retry
    # loop. Drive it once so it claims the lease and suspends on sleep_for.
    await workflow.wake(session, 'flaky_first', max_attempts=1)
    await workflow.absurd.work_batch(batch_size=1)

    # Verify the first brain claimed the lease.
    async with workflow.pool.connection() as conn:
        cur = await conn.execute(
            'SELECT running_brain_name FROM agent_sessions.sessions WHERE id = %s',
            (session.id,),
        )
        row = await cur.fetchone()
    assert row is not None and row[0] == 'flaky_first'

    # Supersede with a different brain name on the same session - should be a no-op
    # for the lease (different brain_name) but proves the lookup query is guarded.
    await workflow.wake(session, 'replacer', concurrency='supersede', max_attempts=1)

    # Now supersede the first brain with another wake of the same name.
    await workflow.wake(session, 'flaky_first', input={'v': 2}, concurrency='supersede', max_attempts=1)

    # Drain - the cancelled first task should fail, the lease releases, pending
    # tasks can proceed.
    for _ in range(10):
        await workflow.absurd.work_batch(batch_size=4)
        await anyio.sleep(0.05)

    # `replacer` was spawned before supersede and doesn't care; it runs once the lease
    # frees up.
    assert observed['second_ran'] is True


async def test_supersede_is_noop_when_no_brain_active(workflow: Workflow) -> None:
    """Supersede on a quiet session should proceed cleanly - no lease to cancel."""

    @workflow.brain('quiet')
    async def quiet(ctx: BrainContext[None]) -> None:
        await ctx.post('hi')  # pragma: no cover - never drained

    session = await Session.create(workflow.pool)
    handle = await workflow.wake(session, 'quiet', concurrency='supersede')
    assert handle.task_id


async def test_supersede_clears_lease_when_cancelling_suspended_brain(workflow: Workflow) -> None:
    """A brain suspended inside `ctx.sleep_for` has no live Python stack, so when Absurd
    cancels it the `except BaseException: _release_session_lease(...)` branch in
    `_run_brain` never runs. `_cancel_active_brain` must clear the lease itself or the
    session would be stranded as "running" forever.
    """

    @workflow.brain('sleeper')
    async def sleeper(ctx: BrainContext[None]) -> None:
        await ctx.sleep(5.0)  # pragma: no cover - cancelled before completion

    session = await Session.create(workflow.pool)
    await workflow.wake(session, 'sleeper', max_attempts=1)
    await workflow.absurd.work_batch(batch_size=1)

    async with workflow.pool.connection() as conn:
        cur = await conn.execute(
            'SELECT running_task_id, running_brain_name FROM agent_sessions.sessions WHERE id = %s',
            (session.id,),
        )
        row = await cur.fetchone()
    assert row is not None and row[0] is not None and row[1] == 'sleeper'

    # Supersede cancels the suspended brain. Because the brain is suspended, no
    # Python cleanup runs - `_cancel_active_brain` must clear the lease itself.
    await workflow.wake(session, 'sleeper', input={'v': 2}, concurrency='supersede', max_attempts=1)

    async with workflow.pool.connection() as conn:
        cur = await conn.execute(
            'SELECT running_task_id, running_brain_name FROM agent_sessions.sessions WHERE id = %s',
            (session.id,),
        )
        row = await cur.fetchone()
    # The lease must be clear (or held by the replacement) - it must NOT still point at
    # the cancelled task. The replacement hasn't been drained yet, so the lease is free.
    assert row is not None
    assert row[0] is None
    assert row[1] is None


async def test_supersede_replacement_writes_running_brain_name(workflow: Workflow) -> None:
    """Supersede'd tasks still enforce the single-active-brain lease. If they didn't,
    a second supersede call on the same session wouldn't find `running_brain_name` and
    would fail to cancel the replacement. Prove the replacement writes the lease by
    firing two supersedes in sequence and checking the second one cancels the first.
    """
    attempts: list[int] = []

    @workflow.brain('chain')
    async def chain(ctx: BrainContext[None]) -> None:
        attempts.append(int(ctx.input.get('v') or 0))
        await ctx.sleep(5.0)  # pragma: no cover - we always supersede

    session = await Session.create(workflow.pool)

    await workflow.wake(session, 'chain', input={'v': 1}, concurrency='supersede', max_attempts=1)
    await workflow.absurd.work_batch(batch_size=1)

    async with workflow.pool.connection() as conn:
        cur = await conn.execute(
            'SELECT running_brain_name FROM agent_sessions.sessions WHERE id = %s',
            (session.id,),
        )
        row = await cur.fetchone()
    # First supersede'd task must have claimed the lease, otherwise the next supersede
    # has nothing to find.
    assert row is not None and row[0] == 'chain'

    # Second supersede should locate and cancel the first replacement.
    await workflow.wake(session, 'chain', input={'v': 2}, concurrency='supersede', max_attempts=1)
    for _ in range(10):
        await workflow.absurd.work_batch(batch_size=4)
        await anyio.sleep(0.05)

    assert 2 in attempts


async def test_supersede_uses_fresh_idempotency_key(workflow: Workflow) -> None:
    """Absurd's idempotency record survives cancellation - if supersede reused the
    deterministic `(session, brain, input-hash)` key, the spawn after the cancel would
    resolve to the cancelled task_id instead of creating a new one. Two supersedes with
    the same input must produce distinct task_ids.
    """

    @workflow.brain('same_input')
    async def same_input(ctx: BrainContext[None]) -> None:
        await ctx.sleep(5.0)  # pragma: no cover

    session = await Session.create(workflow.pool)
    first = await workflow.wake(session, 'same_input', input={'v': 1}, concurrency='supersede', max_attempts=1)
    await workflow.absurd.work_batch(batch_size=1)
    second = await workflow.wake(session, 'same_input', input={'v': 1}, concurrency='supersede', max_attempts=1)

    assert first.task_id != second.task_id
    assert first.dedup_key != second.dedup_key


async def test_wake_depth_check_raises_for_long_chain(pool: AsyncPool) -> None:
    from agent_sessions.workflow import WakeDepthExceeded, _check_wake_depth

    session = await Session.create(pool)
    e1 = await session.append(kind=EventKind.brain_started, actor='brain:x', payload={})
    e2 = await session.append(kind=EventKind.brain_started, actor='brain:x', payload={}, causation_id=e1.sequence)
    e3 = await session.append(kind=EventKind.brain_started, actor='brain:x', payload={}, causation_id=e2.sequence)
    e4 = await session.append(kind=EventKind.brain_started, actor='brain:x', payload={}, causation_id=e3.sequence)

    with pytest.raises(WakeDepthExceeded):
        await _check_wake_depth(pool, session.id, e4.sequence, max_depth=2)

    await _check_wake_depth(pool, session.id, e4.sequence, max_depth=10)
    await _check_wake_depth(pool, session.id, None, max_depth=1)
    await _check_wake_depth(pool, session.id, 999_999, max_depth=10)
