from __future__ import annotations

import anyio
import pytest
from absurd_sdk import AsyncAbsurd, JsonValue
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_absurd import AbsurdAgent

from agent_sessions import (
    BrainContext,
    EventKind,
    Session,
    brain,
    clear_registry,
    create_worker,
    registered_brains,
    wake,
)

from .conftest import AsyncPool

pytestmark = pytest.mark.anyio


def test_brain_decorator_registers_once() -> None:
    @brain('x')
    async def _brain_x(ctx: BrainContext[None]) -> None:
        pass  # pragma: no cover - not driven here

    assert [d.name for d in registered_brains()] == ['x']


def test_duplicate_brain_raises() -> None:
    @brain('dup')
    async def _brain_a(ctx: BrainContext[None]) -> None:  # pragma: no cover
        pass

    with pytest.raises(ValueError, match='already registered'):

        @brain('dup')
        async def _brain_b(ctx: BrainContext[None]) -> None:  # pragma: no cover
            pass


async def test_brain_runs_and_emits_lifecycle(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('greeter')
    async def greeter_brain(ctx: BrainContext[None]) -> None:
        await ctx.post('hello from greeter')

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    handle = await wake(absurd, session, 'greeter')
    assert not handle.deduplicated

    await absurd.work_batch(batch_size=1)

    kinds = [e.kind for e in await session.events()]
    assert kinds == [
        EventKind.brain_started,
        EventKind.assistant_message,
        EventKind.brain_finished,
    ]


async def test_brain_failure_emits_brain_failed(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('bad')
    async def bad_brain(ctx: BrainContext[None]) -> None:
        raise RuntimeError('boom')

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    await wake(absurd, session, 'bad', max_attempts=1)

    await absurd.work_batch(batch_size=1)

    events = await session.events()
    kinds = [e.kind for e in events]
    assert EventKind.brain_failed in kinds
    failed = next(e for e in events if e.kind == EventKind.brain_failed)
    assert 'boom' in failed.payload['error']


async def test_brain_on_poison_handler_is_called(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    captured = []

    async def on_poison(event: object) -> None:
        captured.append(event)

    @brain('bad')
    async def bad_brain(ctx: BrainContext[None]) -> None:
        raise RuntimeError('kaboom')

    await create_worker(absurd=absurd, pool=pool, on_poison=on_poison)
    session = await Session.create(pool)
    await wake(absurd, session, 'bad', max_attempts=1)
    await absurd.work_batch(batch_size=1)

    assert len(captured) == 1


async def test_post_status_emits_status_update(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('status')
    async def status_brain(ctx: BrainContext[None]) -> None:
        await ctx.post_status('starting')

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    await wake(absurd, session, 'status')
    await absurd.work_batch(batch_size=1)

    kinds = [e.kind for e in await session.events()]
    assert EventKind.status_update in kinds


async def test_brain_chain_via_ctx_wake(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('router')
    async def router(ctx: BrainContext[None]) -> None:
        await ctx.post('routing')
        await ctx.wake('analyst')

    @brain('analyst')
    async def analyst(ctx: BrainContext[None]) -> None:
        await ctx.post('analyzed')

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    await wake(absurd, session, 'router')

    # Drain: router runs (spawns analyst), then analyst runs.
    for _ in range(3):
        await absurd.work_batch(batch_size=2)

    events = await session.events()
    assistant = [e for e in events if e.kind == EventKind.assistant_message]
    contents = [e.payload['content'] for e in assistant]
    assert contents == ['routing', 'analyzed']


async def test_brain_registry_is_cleared_between_tests() -> None:
    assert registered_brains() == []
    clear_registry()
    assert registered_brains() == []


async def test_brain_input_is_accessible(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    observed: dict[str, object] = {}

    @brain('echo')
    async def echo_brain(ctx: BrainContext[None]) -> None:
        observed['input'] = dict(ctx.input)
        observed['name'] = ctx.name
        observed['session_id'] = ctx.session.id
        observed['absurd_ctx'] = ctx.absurd_ctx

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    await wake(absurd, session, 'echo', input={'msg': 'hi'})
    await absurd.work_batch(batch_size=1)

    assert observed['input'] == {'msg': 'hi'}
    assert observed['name'] == 'echo'
    assert observed['session_id'] == session.id
    assert observed['absurd_ctx'] is not None


async def test_agent_run_threads_history_and_appends_messages(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    seen_history: list[list[ModelMessage]] = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_history.append(list(messages))
        return ModelResponse(parts=[TextPart(content='pong')])

    model = FunctionModel(fn, model_name='m')
    inner_agent = Agent(model, name='inner')
    absurd_agent = AbsurdAgent(inner_agent, absurd, name='inner')

    @brain('driver')
    async def driver(ctx: BrainContext[None]) -> None:
        await ctx.agent_run(absurd_agent, 'continue')
        # Second agent_run sees the first's response via the session history.
        await ctx.agent_run(absurd_agent, 'again')

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    await wake(absurd, session, 'driver')
    await absurd.work_batch(batch_size=1)

    # Two model calls; the second run received the first run's messages as history.
    assert len(seen_history) == 2
    assert any(isinstance(m, ModelResponse) for m in seen_history[1])


async def test_brain_sleep_checkpoints_via_absurd(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('sleeper')
    async def sleeper(ctx: BrainContext[None]) -> None:
        await ctx.sleep(0.1)
        await ctx.post('awake')

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    await wake(absurd, session, 'sleeper')
    await absurd.work_batch(batch_size=1)  # suspends on sleep_for
    await anyio.sleep(0.3)
    await absurd.work_batch(batch_size=1)  # resumes

    contents = [e.payload.get('content') for e in await session.events() if e.kind == EventKind.assistant_message]
    assert 'awake' in contents


async def test_worker_register_is_idempotent(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('once')
    async def once_brain(ctx: BrainContext[None]) -> None:  # pragma: no cover
        pass

    worker = await create_worker(absurd=absurd, pool=pool)
    worker.register()  # second call hits the `_registered` short-circuit
    worker.register()


async def test_worker_rejects_non_object_input(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('needs_object')
    async def _br(ctx: BrainContext[None]) -> None:  # pragma: no cover - never reached
        pass

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    # Pretend an upstream caller spawned directly with a non-object input; the
    # brain handler should fail the task.
    params: JsonValue = {'session_id': str(session.id), 'input': 'not-a-dict'}
    await absurd.spawn('brain.needs_object', params, max_attempts=1)
    await absurd.work_batch(batch_size=1)
    # Task should be failed - we don't verify the error text, just that the
    # code path executed without crashing the worker loop.


async def test_worker_parallel_concurrency_skips_lock(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('par')
    async def par_brain(ctx: BrainContext[None]) -> None:
        await ctx.post('parallel')

    await create_worker(absurd=absurd, pool=pool)
    session = await Session.create(pool)
    await wake(absurd, session, 'par', concurrency='parallel')
    await absurd.work_batch(batch_size=1)
    contents = [e.payload.get('content') for e in await session.events() if e.kind == EventKind.assistant_message]
    assert 'parallel' in contents


async def test_worker_missing_session_id_fails_task(pool: AsyncPool, absurd: AsyncAbsurd) -> None:
    @brain('needs_id')
    async def _br(ctx: BrainContext[None]) -> None:  # pragma: no cover
        pass

    await create_worker(absurd=absurd, pool=pool)
    # Spawn with missing session_id; the _expect_uuid validation raises.
    await absurd.spawn('brain.needs_id', {'input': {}}, max_attempts=1)
    await absurd.work_batch(batch_size=1)


async def test_wake_depth_check_raises_for_long_chain(pool: AsyncPool) -> None:
    from agent_sessions.worker import WakeDepthExceeded, _check_wake_depth

    session = await Session.create(pool)
    # Build a chain of brain_started events: 1 <- 2 <- 3 <- 4
    e1 = await session.append(kind=EventKind.brain_started, actor='brain:x', payload={})
    e2 = await session.append(kind=EventKind.brain_started, actor='brain:x', payload={}, causation_id=e1.sequence)
    e3 = await session.append(kind=EventKind.brain_started, actor='brain:x', payload={}, causation_id=e2.sequence)
    e4 = await session.append(kind=EventKind.brain_started, actor='brain:x', payload={}, causation_id=e3.sequence)

    # From e4, chasing causation reaches e3 -> e2 -> e1 -> None: depth 3.
    with pytest.raises(WakeDepthExceeded):
        await _check_wake_depth(pool, session.id, e4.sequence, max_depth=2)

    # With a big max, no exception.
    await _check_wake_depth(pool, session.id, e4.sequence, max_depth=10)
    # causation_id=None short-circuits.
    await _check_wake_depth(pool, session.id, None, max_depth=1)
    # Missing-event short-circuits (return inside the loop).
    await _check_wake_depth(pool, session.id, 999_999, max_depth=10)
