from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

import pytest
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from pydantic_ai import Agent, ModelMessage, ModelResponse
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_absurd import AbsurdDurability

from .conftest import reenter_running_task, running_task_context

pytestmark = pytest.mark.anyio


def _make_model(counter: dict[str, int] | None = None) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if counter is not None:
            counter['calls'] += 1
        return ModelResponse(parts=[TextPart(content='ok')])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        if counter is not None:
            counter['calls'] += 1
        yield 'ok'

    return FunctionModel(fn, stream_function=stream_fn, model_name='fn')


async def _register_noop(absurd: AsyncAbsurd, name: str = 'noop') -> None:
    async def noop(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:  # pragma: no cover
        return None

    absurd.register_task(name=name)(noop)


async def test_requires_name() -> None:
    with pytest.raises(UserError, match='unique `name`'):
        Agent(_make_model(), capabilities=[AbsurdDurability()])


async def test_name_from_capability() -> None:
    agent = Agent(_make_model(), capabilities=[AbsurdDurability(name='custom')])
    bound = AbsurdDurability.from_agent(agent)
    assert bound is not None
    assert bound.name == 'custom'


async def test_requires_model() -> None:
    with pytest.raises(UserError, match='needs to have a `model`'):
        Agent(name='a', capabilities=[AbsurdDurability()])


async def test_reserved_default_model_id_raises() -> None:
    with pytest.raises(UserError, match="'default' is reserved"):
        Agent(_make_model(), name='a', capabilities=[AbsurdDurability(models={'default': _make_model()})])


async def test_leaf_toolset_without_id_is_durable(absurd: AsyncAbsurd) -> None:
    """An id-less toolset keeps working (and keeps the wrapper's step names) under the capability."""
    tool_calls = {'calls': 0}
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def charge_card(amount: int) -> str:
        tool_calls['calls'] += 1
        return f'charged {amount}'

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='charge_card', args={'amount': 7})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(
        FunctionModel(fn, model_name='fn'),
        name='idless',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    await _register_noop(absurd, 'idless')

    async with running_task_context(absurd, 'idless', max_attempts=2) as ctx:
        first = await agent.run('charge it')
        task_id = ctx.task_id

    async with reenter_running_task(absurd, task_id):
        replayed = await agent.run('charge it')

    assert tool_calls['calls'] == 1
    assert replayed.output == first.output == 'done'


async def test_same_toolset_instance_in_two_places_is_wrapped_once() -> None:
    toolset = FunctionToolset[None](id='shared')

    @toolset.tool_plain
    def echo(value: str) -> str:  # pragma: no cover - never invoked, only wrap check
        return value

    agent = Agent(_make_model(), name='a', toolsets=[toolset, toolset], capabilities=[AbsurdDurability()])
    bound = AbsurdDurability.from_agent(agent)
    assert bound is not None
    # One wrapper for `toolset`, one for the agent's own `<agent>` toolset.
    assert len(bound._wrappers_by_leaf) == 2


async def test_duplicate_toolset_id_raises() -> None:
    first = FunctionToolset[None](id='tools')

    @first.tool_plain
    def echo(value: str) -> str:  # pragma: no cover - never invoked, only wrap check
        return value

    second = FunctionToolset[None](id='tools')

    @second.tool_plain
    def shout(value: str) -> str:  # pragma: no cover - never invoked, only wrap check
        return value.upper()

    with pytest.raises(UserError, match='same `id`'):
        Agent(_make_model(), name='a', toolsets=[first, second], capabilities=[AbsurdDurability()])


async def test_from_agent_without_capability_returns_none() -> None:
    agent = Agent(_make_model(), name='a')
    assert AbsurdDurability.from_agent(agent) is None


async def test_from_agent_multiple_raises() -> None:
    agent = Agent(_make_model(), name='a', capabilities=[AbsurdDurability(), AbsurdDurability()])
    with pytest.raises(UserError, match='at most one'):
        AbsurdDurability.from_agent(agent)


async def test_run_outside_task_is_transparent() -> None:
    counter = {'calls': 0}
    agent = Agent(_make_model(counter), name='a', capabilities=[AbsurdDurability()])
    result = await agent.run('hi')
    assert result.output == 'ok'
    assert counter['calls'] == 1


async def test_run_inside_task_completes(absurd: AsyncAbsurd) -> None:
    agent = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
    await _register_noop(absurd)
    async with running_task_context(absurd, 'noop'):
        result = await agent.run('hi')
    assert result.output == 'ok'


async def test_run_inside_authored_task_is_durable(absurd: AsyncAbsurd) -> None:
    agent = Agent(_make_model(), name='analyst', capabilities=[AbsurdDurability()])

    async def analyse(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
        assert isinstance(params, dict)
        prompt = params['prompt']
        assert isinstance(prompt, str)
        result = await agent.run(prompt)
        return {'output': result.output}

    absurd.register_task(name='analyse')(analyse)

    spawned = await absurd.spawn('analyse', {'prompt': 'go'})
    await absurd.work_batch(batch_size=1)
    result = await absurd.fetch_task_result(spawned['task_id'])
    assert result is not None and result.state == 'completed'
    assert result.result == {'output': 'ok'}


async def test_replay_serves_cached_model_response(absurd: AsyncAbsurd) -> None:
    counter = {'calls': 0}
    agent = Agent(_make_model(counter), name='crash', capabilities=[AbsurdDurability()])
    await _register_noop(absurd, 'crash')

    async with running_task_context(absurd, 'crash', max_attempts=2) as ctx:
        first = await agent.run('hi')
        task_id = ctx.task_id

    async with reenter_running_task(absurd, task_id):
        replayed = await agent.run('hi')

    assert counter['calls'] == 1
    assert replayed.output == first.output == 'ok'


async def test_replay_does_not_rerun_function_tool(absurd: AsyncAbsurd) -> None:
    tool_calls = {'calls': 0}
    toolset = FunctionToolset[None](id='tools')

    @toolset.tool_plain
    def charge_card(amount: int) -> str:
        tool_calls['calls'] += 1
        return f'charged {amount}'

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='charge_card', args={'amount': 42})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(
        FunctionModel(fn, model_name='fn'),
        name='billing',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )
    await _register_noop(absurd, 'billing')

    async with running_task_context(absurd, 'billing', max_attempts=2) as ctx:
        first = await agent.run('charge it')
        task_id = ctx.task_id

    async with reenter_running_task(absurd, task_id):
        replayed = await agent.run('charge it')

    assert tool_calls['calls'] == 1
    assert replayed.output == first.output == 'done'


async def test_registered_model_selected_per_run(absurd: AsyncAbsurd) -> None:
    primary = {'calls': 0}
    cheap = {'calls': 0}

    def primary_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        primary['calls'] += 1
        return ModelResponse(parts=[TextPart(content='primary')])

    def cheap_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        cheap['calls'] += 1
        return ModelResponse(parts=[TextPart(content='cheap')])

    agent = Agent(
        FunctionModel(primary_fn, model_name='primary'),
        name='a',
        capabilities=[AbsurdDurability(models={'cheap': FunctionModel(cheap_fn, model_name='cheap')})],
    )
    await _register_noop(absurd)

    async with running_task_context(absurd, 'noop'):
        default_result = await agent.run('hi')
        cheap_result = await agent.run('hi', model='cheap')

    assert default_result.output == 'primary'
    assert cheap_result.output == 'cheap'
    assert primary['calls'] == 1
    assert cheap['calls'] == 1


async def test_runtime_function_toolset_rejected(absurd: AsyncAbsurd) -> None:
    agent: Agent[None, str] = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
    toolset = FunctionToolset[None](id='late')

    @toolset.tool_plain
    def echo(value: str) -> str:  # pragma: no cover - rejected before it can run
        return value

    await _register_noop(absurd)
    async with running_task_context(absurd, 'noop'):
        with pytest.raises(UserError, match='cannot be passed to `run\\(toolsets=...\\)` at runtime'):
            await agent.run('hi', toolsets=[toolset])


def _tool_calling_model(tool_name: str) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args={})])
        return ModelResponse(parts=[TextPart(content='done')])

    return FunctionModel(fn, model_name='fn')


def _late_toolset(calls: dict[str, int]) -> FunctionToolset[None]:
    toolset = FunctionToolset[None](id='late')

    @toolset.tool_plain
    def late() -> str:
        calls['calls'] += 1
        return 'late result'

    return toolset


async def test_override_toolsets_rejected_inside_task(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}
    agent: Agent[None, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[AbsurdDurability()])
    await _register_noop(absurd)
    async with running_task_context(absurd, 'noop'):
        with agent.override(toolsets=[_late_toolset(calls)]):
            with pytest.raises(UserError, match='cannot be passed to `run\\(toolsets=...\\)` at runtime'):
                await agent.run('hi')
    assert calls['calls'] == 0


async def test_override_toolsets_respected_outside_task() -> None:
    calls = {'calls': 0}
    agent: Agent[None, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[AbsurdDurability()])
    with agent.override(toolsets=[_late_toolset(calls)]):
        result = await agent.run('hi')
    assert result.output == 'done'
    assert calls['calls'] == 1


async def test_override_tools_rejected_inside_task(absurd: AsyncAbsurd) -> None:
    calls = {'calls': 0}

    def late() -> str:  # pragma: no cover - rejected before it can run
        calls['calls'] += 1
        return 'late result'

    agent: Agent[None, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[AbsurdDurability()])
    await _register_noop(absurd)
    async with running_task_context(absurd, 'noop'):
        with agent.override(tools=[late]):
            with pytest.raises(UserError, match='cannot be passed to `run\\(toolsets=...\\)` at runtime'):
                await agent.run('hi')
    assert calls['calls'] == 0


async def test_override_tools_respected_outside_task() -> None:
    """The overriding toolset shares the `<agent>` id; instance-keyed wrapping must not swap it away."""
    calls = {'calls': 0}

    def late() -> str:
        calls['calls'] += 1
        return 'late result'

    agent: Agent[None, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[AbsurdDurability()])
    with agent.override(tools=[late]):
        result = await agent.run('hi')
    assert result.output == 'done'
    assert calls['calls'] == 1


async def test_capability_owned_toolset_is_durable(absurd: AsyncAbsurd) -> None:
    """A toolset contributed by a capability is registered at construction, so it must be
    wrapped and checkpointed, not rejected as a runtime toolset because of the
    `CapabilityOwnedToolset` wrapper Pydantic AI puts around it."""
    tool_calls = {'calls': 0}
    toolset = FunctionToolset[None](id='owned')

    @toolset.tool_plain
    def charge_card(amount: int) -> str:
        tool_calls['calls'] += 1
        return f'charged {amount}'

    class DemoCapability(AbstractCapability[None]):
        def get_toolset(self) -> FunctionToolset[None]:
            return toolset

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='charge_card', args={'amount': 5})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent: Agent[None, str] = Agent(
        FunctionModel(fn, model_name='fn'),
        name='owner',
        capabilities=[DemoCapability(), AbsurdDurability()],
    )
    await _register_noop(absurd, 'owner')

    async with running_task_context(absurd, 'owner', max_attempts=2) as ctx:
        first = await agent.run('charge it')
        task_id = ctx.task_id

    async with reenter_running_task(absurd, task_id):
        replayed = await agent.run('charge it')

    assert tool_calls['calls'] == 1
    assert replayed.output == first.output == 'done'


async def test_runtime_toolset_still_rejected_alongside_capability_toolset(absurd: AsyncAbsurd) -> None:
    """Ignoring wrapper nodes must not make genuine runtime toolsets slip through."""
    owned = FunctionToolset[None](id='owned')

    @owned.tool_plain
    def greet() -> str:  # pragma: no cover - never invoked
        return 'hello'

    class DemoCapability(AbstractCapability[None]):
        def get_toolset(self) -> FunctionToolset[None]:
            return owned

    agent: Agent[None, str] = Agent(_make_model(), name='a', capabilities=[DemoCapability(), AbsurdDurability()])
    await _register_noop(absurd)
    async with running_task_context(absurd, 'noop'):
        with pytest.raises(UserError, match='cannot be passed to `run\\(toolsets=...\\)` at runtime'):
            await agent.run('hi', toolsets=[_late_toolset({'calls': 0})])


async def test_runtime_external_toolset_allowed(absurd: AsyncAbsurd) -> None:
    agent: Agent[None, str] = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
    await _register_noop(absurd)
    async with running_task_context(absurd, 'noop'):
        result = await agent.run('hi', toolsets=[ExternalToolset[None](tool_defs=[])])
    assert result.output == 'ok'


async def test_construction_external_toolset_passes_through_unwrapped() -> None:
    external = ExternalToolset[None](tool_defs=[])
    agent = Agent(_make_model(), name='a', toolsets=[external], capabilities=[AbsurdDurability()])
    assert any(t is external for t in agent.toolsets)


async def test_mcp_tool_call_inside_task(absurd: AsyncAbsurd) -> None:
    from fastmcp import FastMCP
    from pydantic_ai.mcp import MCPToolset

    server: FastMCP[None] = FastMCP(name='calc')

    @server.tool
    def add(a: int, b: int) -> int:
        return a + b

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='add', args={'a': 2, 'b': 3})])
        return ModelResponse(parts=[TextPart(content='summed')])

    agent = Agent(
        FunctionModel(fn, model_name='fn'),
        name='calc',
        toolsets=[MCPToolset[None](server, id='calc')],
        capabilities=[AbsurdDurability()],
    )
    await _register_noop(absurd)

    async with running_task_context(absurd, 'noop'):
        result = await agent.run('add 2 and 3')
    assert result.output == 'summed'


async def test_event_stream_handler_receives_events(absurd: AsyncAbsurd) -> None:
    events: list[AgentStreamEvent] = []

    async def handler(run_ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='greet', json_args='{}')}
        else:
            yield 'done'

    toolset = FunctionToolset[None](id='tools')

    @toolset.tool_plain
    def greet() -> str:
        return 'hello'

    agent = Agent(
        FunctionModel(stream_function=stream_fn, model_name='fn'),
        name='a',
        toolsets=[toolset],
        capabilities=[AbsurdDurability(event_stream_handler=handler)],
    )
    await _register_noop(absurd)

    async with running_task_context(absurd, 'noop'):
        result = await agent.run('hi')

    assert result.output == 'done'
    assert any(isinstance(e, PartStartEvent | PartDeltaEvent) for e in events)
    assert any(isinstance(e, FunctionToolCallEvent) for e in events)


async def test_run_stream_inside_task_replays_buffered_stream(absurd: AsyncAbsurd) -> None:
    counter = {'calls': 0}
    agent = Agent(_make_model(counter), name='a', capabilities=[AbsurdDurability()])
    await _register_noop(absurd)
    async with running_task_context(absurd, 'noop'):
        async with agent.run_stream('hi') as result:
            assert await result.get_output() == 'ok'
    assert counter['calls'] == 1


async def test_run_stream_events_inside_task(absurd: AsyncAbsurd) -> None:
    agent = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
    await _register_noop(absurd)
    async with running_task_context(absurd, 'noop'):
        async with agent.run_stream_events('hi') as stream:
            events = [event async for event in stream]
    assert any(isinstance(e, PartStartEvent) for e in events)


async def test_iter_inside_task(absurd: AsyncAbsurd) -> None:
    agent = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
    await _register_noop(absurd)
    async with running_task_context(absurd, 'noop'):
        async with agent.iter('hi') as run:
            async for _ in run:
                pass
    assert run.result is not None
    assert run.result.output == 'ok'


async def test_wrapper_written_stream_checkpoint_replays_under_capability(absurd: AsyncAbsurd) -> None:
    """A `request_stream` checkpoint recorded by the deprecated `AbsurdAgent` wrapper
    (a bare `ModelResponse`) is replayed correctly by the capability path."""
    counter = {'calls': 0}
    agent = Agent(_make_model(counter), name='legacy', capabilities=[AbsurdDurability()])
    await _register_noop(absurd, 'legacy')

    async with running_task_context(absurd, 'legacy', max_attempts=2) as ctx:
        legacy_payload = ModelResponse(parts=[TextPart(content='from-wrapper')])
        from pydantic_ai_absurd._model import _serialize

        async def _write_legacy() -> dict[str, JsonValue]:
            return _serialize(legacy_payload)

        await ctx.step('legacy__model.request_stream', _write_legacy)
        task_id = ctx.task_id

    async with reenter_running_task(absurd, task_id):
        async with agent.run_stream('hi') as result:
            assert await result.get_output() == 'from-wrapper'

    assert counter['calls'] == 0


async def test_string_default_model_replays_wrapper_checkpoint(absurd: AsyncAbsurd) -> None:
    """A string default model checkpoints under the suffix-less step name the wrapper used,
    so wrapper-era checkpoints replay after migrating."""
    agent = Agent('test', name='strdef', capabilities=[AbsurdDurability()])
    await _register_noop(absurd, 'strdef')

    async with running_task_context(absurd, 'strdef', max_attempts=2) as ctx:
        legacy_payload = ModelResponse(parts=[TextPart(content='from-wrapper')])
        from pydantic_ai_absurd._model import _serialize

        async def _write_legacy() -> dict[str, JsonValue]:
            return _serialize(legacy_payload)

        await ctx.step('strdef__model.request', _write_legacy)
        task_id = ctx.task_id

    async with reenter_running_task(absurd, task_id):
        replayed = await agent.run('hi')

    assert replayed.output == 'from-wrapper'


async def test_cancel_suspended_response_is_checkpointed(absurd: AsyncAbsurd) -> None:
    cancelled: list[ModelResponse] = []

    class CancellableModel(FunctionModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response)

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
        return ModelResponse(parts=[TextPart(content='ok')])

    model = CancellableModel(fn, model_name='fn')
    agent = Agent(model, name='a', capabilities=[AbsurdDurability()])
    bound = AbsurdDurability.from_agent(agent)
    assert bound is not None

    ctx = RunContext[None](deps=None, model=model, usage=RunUsage())
    request_context = ModelRequestContext(
        model=model, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    response = ModelResponse(parts=[TextPart(content='suspended')])

    async def handler(request: ModelRequestContext) -> ModelResponse:
        await request.model.cancel_suspended_response(response)
        return response

    await _register_noop(absurd)
    async with running_task_context(absurd, 'noop'):
        result = await bound.wrap_model_request(ctx, request_context=request_context, handler=handler)

    assert result is response
    assert cancelled == [response]
