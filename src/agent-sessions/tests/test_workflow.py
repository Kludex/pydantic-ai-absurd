from __future__ import annotations

import anyio
import pytest
from absurd_sdk import AsyncAbsurd, JsonValue
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

    workflow.register()
    session = await Session.create(workflow.pool)
    handle = await workflow.wake(session, 'greeter')
    assert not handle.deduplicated

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

    workflow.register()
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

    workflow.register()
    session = await Session.create(pool)
    await workflow.wake(session, 'bad', max_attempts=1)
    await absurd.work_batch(batch_size=1)

    assert len(captured) == 1


async def test_post_status_emits_status_update(workflow: Workflow) -> None:
    @workflow.brain('status')
    async def status_brain(ctx: BrainContext[None]) -> None:
        await ctx.post_status('starting')

    workflow.register()
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

    workflow.register()
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

    workflow.register()
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

    workflow.register()
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

    workflow.register()
    session = await Session.create(workflow.pool)
    await workflow.wake(session, 'sleeper')
    await workflow.absurd.work_batch(batch_size=1)  # suspends on sleep_for
    await anyio.sleep(0.3)
    await workflow.absurd.work_batch(batch_size=1)  # resumes

    contents = [e.payload.get('content') for e in await session.events() if e.kind == EventKind.assistant_message]
    assert 'awake' in contents


async def test_register_is_idempotent(workflow: Workflow) -> None:
    @workflow.brain('once')
    async def once_brain(ctx: BrainContext[None]) -> None:  # pragma: no cover
        pass

    workflow.register()
    workflow.register()
    workflow.register()


async def test_rejects_non_object_input(workflow: Workflow) -> None:
    @workflow.brain('needs_object')
    async def _br(ctx: BrainContext[None]) -> None:  # pragma: no cover - never reached
        pass

    workflow.register()
    session = await Session.create(workflow.pool)
    params: JsonValue = {'session_id': str(session.id), 'input': 'not-a-dict'}
    await workflow.absurd.spawn('brain.needs_object', params, max_attempts=1)
    await workflow.absurd.work_batch(batch_size=1)


async def test_parallel_concurrency_skips_lock(workflow: Workflow) -> None:
    @workflow.brain('par')
    async def par_brain(ctx: BrainContext[None]) -> None:
        await ctx.post('parallel')

    workflow.register()
    session = await Session.create(workflow.pool)
    await workflow.wake(session, 'par', concurrency='parallel')
    await workflow.absurd.work_batch(batch_size=1)
    contents = [e.payload.get('content') for e in await session.events() if e.kind == EventKind.assistant_message]
    assert 'parallel' in contents


async def test_missing_session_id_fails_task(workflow: Workflow) -> None:
    @workflow.brain('needs_id')
    async def _br(ctx: BrainContext[None]) -> None:  # pragma: no cover
        pass

    workflow.register()
    await workflow.absurd.spawn('brain.needs_id', {'input': {}}, max_attempts=1)
    await workflow.absurd.work_batch(batch_size=1)


async def test_deterministic_dedup_collapses_duplicate_wakes(workflow: Workflow) -> None:
    @workflow.brain('dedup_x')
    async def _br(ctx: BrainContext[None]) -> None:
        await ctx.post('once')  # pragma: no cover - not drained

    workflow.register()
    session = await Session.create(workflow.pool)
    first = await workflow.wake(session, 'dedup_x', input={'k': 'v'})
    second = await workflow.wake(session, 'dedup_x', input={'k': 'v'})
    assert first.task_id == second.task_id
    assert first.deduplicated is False
    assert second.deduplicated is True


async def test_explicit_dedup_key_collapses(workflow: Workflow) -> None:
    @workflow.brain('dedup_y')
    async def _br(ctx: BrainContext[None]) -> None:
        await ctx.post('p')  # pragma: no cover - not drained

    workflow.register()
    session = await Session.create(workflow.pool)
    first = await workflow.wake(session, 'dedup_y', dedup_key='custom', input={'k': 1})
    second = await workflow.wake(session, 'dedup_y', dedup_key='custom', input={'k': 2})
    assert first.task_id == second.task_id


async def test_supersede_cancels_existing_tasks(workflow: Workflow) -> None:
    @workflow.brain('dedup_s')
    async def _br(ctx: BrainContext[None]) -> None:
        pass  # pragma: no cover

    workflow.register()
    session = await Session.create(workflow.pool)
    first = await workflow.wake(session, 'dedup_s', input={'v': 1})
    second = await workflow.wake(session, 'dedup_s', input={'v': 2}, concurrency='supersede')
    assert second.task_id != first.task_id


async def test_wake_by_session_id_only(workflow: Workflow) -> None:
    @workflow.brain('dedup_w')
    async def _br(ctx: BrainContext[None]) -> None:
        pass  # pragma: no cover

    workflow.register()
    session = await Session.create(workflow.pool)
    handle = await workflow.wake(session.id, 'dedup_w')
    assert handle.task_id
    assert not handle.deduplicated


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
