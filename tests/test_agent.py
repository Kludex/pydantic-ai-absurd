from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Mapping

import pytest
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from fastmcp import FastMCP
from pydantic import TypeAdapter
from pydantic_ai import Agent, ModelMessage, ModelResponse
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import AgentStreamEvent, ModelRequest, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset
from pydantic_core import to_jsonable_python

from pydantic_ai_absurd import AbsurdAgent, AbsurdFunctionToolset, AbsurdMCPToolset, AbsurdModel

from .conftest import running_task_context

pytestmark = pytest.mark.anyio

_history_adapter: TypeAdapter[list[ModelMessage]] = TypeAdapter(list[ModelMessage])


def _make_model() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='ok')])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'ok'

    return FunctionModel(fn, stream_function=stream_fn, model_name='fn')


async def test_requires_name(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model())
    with pytest.raises(UserError, match='unique `name`'):
        AbsurdAgent(inner, absurd)


async def test_model_swapped_with_absurd_model(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    assert isinstance(agent.model, AbsurdModel)


async def test_function_toolsets_are_wrapped(absurd: AsyncAbsurd) -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def echo(value: str) -> str:  # pragma: no cover - never invoked, only wrap check
        return value

    inner = Agent(_make_model(), toolsets=[toolset], name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    assert any(isinstance(t, AbsurdFunctionToolset) for t in agent.toolsets)


async def test_other_toolsets_pass_through_unwrapped(absurd: AsyncAbsurd) -> None:
    # A toolset that is neither MCP nor a FunctionToolset is left as-is: we only
    # checkpoint the calls we understand (model, MCP, function tools).
    external = ExternalToolset[None](tool_defs=[])
    inner = Agent(_make_model(), toolsets=[external], name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    assert any(t is external for t in agent.toolsets)


async def test_in_process_mcp_toolsets_are_wrapped(absurd: AsyncAbsurd) -> None:
    server: FastMCP[None] = FastMCP(name='calc')

    @server.tool
    def add(a: int, b: int) -> int:  # pragma: no cover - never invoked, only wrap check
        return a + b

    toolset = MCPToolset[None](server)
    inner = Agent(_make_model(), toolsets=[toolset], name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    assert any(isinstance(t, AbsurdMCPToolset) for t in agent.toolsets)


async def test_iter_rejects_non_absurd_model(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    with pytest.raises(UserError, match='Non-Absurd model cannot be overridden'):
        async with agent.iter('hi', model=_make_model()):
            pass  # pragma: no cover


async def test_run_outside_task_context_raises(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    with pytest.raises(UserError, match='must be called from inside an Absurd task handler'):
        await agent.run('hi')


async def test_run_inside_task_context_completes(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')

    async def noop(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:  # pragma: no cover
        return None

    absurd.register_task(name='noop')(noop)

    async with running_task_context(absurd, 'noop'):
        result = await agent.run('hi')
    assert result.output == 'ok'


async def test_run_sync_is_not_supported(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    with pytest.raises(UserError, match='run_sync\\(\\) is not supported'):
        agent.run_sync('hi')


async def test_run_stream_events_is_not_supported(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    with pytest.raises(UserError, match='run_stream_events'):
        agent.run_stream_events('hi')


async def test_run_stream_inside_task_raises(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')

    async def noop(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:  # pragma: no cover
        return None

    absurd.register_task(name='noop')(noop)

    async with running_task_context(absurd, 'noop'):
        with pytest.raises(UserError, match='run_stream'):
            async with agent.run_stream('hi'):
                pass  # pragma: no cover


async def test_run_stream_outside_task_works(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    async with agent.run_stream('hi') as result:
        assert await result.get_output() == 'ok'


async def test_iter_inside_task_works(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')

    async def noop(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:  # pragma: no cover
        return None

    absurd.register_task(name='noop')(noop)

    async with running_task_context(absurd, 'noop'):
        async with agent.iter('hi') as run:
            async for _ in run:
                pass
        assert run.result is not None
        assert run.result.output == 'ok'


async def test_override_rejects_non_absurd_model(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    with pytest.raises(UserError, match='Non-Absurd model cannot be overridden'):
        with agent.override(model=_make_model()):
            pass  # pragma: no cover


async def test_override_accepts_absurd_model(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    replacement = AbsurdModel(_make_model(), step_name_prefix='a')
    with agent.override(model=replacement):
        assert agent.model is not replacement  # _absurd_overrides still active in run path


async def test_run_rejects_non_absurd_model(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    with pytest.raises(UserError, match='Non-Absurd model cannot be overridden'):
        await agent.run('hi', model=_make_model())


async def test_run_inside_authored_task_is_durable(absurd: AsyncAbsurd) -> None:
    """The intended pattern: author a task, call `agent.run()` inside it, spawn the task.

    Threads `message_history` through and checks the run sees the prior conversation.
    """
    seen: list[list[ModelMessage]] = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = AbsurdAgent(Agent(FunctionModel(fn, model_name='fn'), name='analyst'), absurd, name='analyst')

    async def analyse(params: Mapping[str, JsonValue] | None, ctx: AsyncTaskContext) -> JsonValue:
        params = params or {}
        raw_history = params.get('message_history')
        history = _history_adapter.validate_python(raw_history) if isinstance(raw_history, list) else None
        result = await agent.run('continue', message_history=history)
        return {'output': result.output}

    absurd.register_task(name='analyse')(analyse)

    history = [
        ModelRequest(parts=[UserPromptPart(content='earlier turn')]),
        ModelResponse(parts=[TextPart(content='earlier reply')]),
    ]
    spawned = await absurd.spawn('analyse', {'message_history': to_jsonable_python(history)})
    await absurd.work_batch(batch_size=1)
    result = await absurd.fetch_task_result(spawned['task_id'])
    assert result is not None and result.state == 'completed'
    assert isinstance(result.result, dict)
    assert result.result['output'] == 'ok'

    # The run saw the prior conversation, not a blank history.
    parts = [p for messages in seen for m in messages for p in m.parts]
    assert any(isinstance(p, UserPromptPart) and p.content == 'earlier turn' for p in parts)


async def test_run_without_model_on_wrapped_agent_raises(absurd: AsyncAbsurd) -> None:
    inner = Agent(name='needs-model')  # no model
    with pytest.raises(UserError, match='`model` set at construction'):
        AbsurdAgent(inner, absurd, name='needs-model')


async def test_event_stream_handler_override(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')

    async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:  # pragma: no cover
        return None

    agent = AbsurdAgent(inner, absurd, name='a', event_stream_handler=handler)
    assert agent.event_stream_handler is handler


async def test_event_stream_handler_falls_through_to_wrapped(absurd: AsyncAbsurd) -> None:
    inner = Agent(_make_model(), name='a')
    agent = AbsurdAgent(inner, absurd, name='a')
    assert agent.event_stream_handler is None
