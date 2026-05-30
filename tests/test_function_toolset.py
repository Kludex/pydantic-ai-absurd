from __future__ import annotations

import pytest
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from pydantic_ai import FunctionToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_absurd import AbsurdFunctionToolset

from .conftest import reenter_running_task, running_task_context

pytestmark = pytest.mark.anyio


async def _noop(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:  # pragma: no cover
    return None


def _make_toolset(calls: list[str]) -> FunctionToolset[None]:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def shout(value: str) -> str:
        calls.append(value)
        return value.upper()

    return toolset


def _run_context() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt='x', messages=[])


async def test_id_passthrough(absurd: AsyncAbsurd) -> None:
    inner = _make_toolset([])
    toolset = AbsurdFunctionToolset(inner, step_name_prefix='a')
    assert toolset.id == inner.id


async def test_aenter_aexit(absurd: AsyncAbsurd) -> None:
    toolset = AbsurdFunctionToolset(_make_toolset([]), step_name_prefix='a')
    async with toolset as entered:
        assert entered is toolset


async def test_visit_and_replace_returns_self(absurd: AsyncAbsurd) -> None:
    toolset = AbsurdFunctionToolset(_make_toolset([]), step_name_prefix='a')

    def visitor(t: FunctionToolset[None]) -> FunctionToolset[None]:  # pragma: no cover
        return t

    assert toolset.visit_and_replace(visitor) is toolset


async def test_call_tool_without_context_passes_through(absurd: AsyncAbsurd) -> None:
    calls: list[str] = []
    inner = _make_toolset(calls)
    toolset = AbsurdFunctionToolset(inner, step_name_prefix='a')
    ctx = _run_context()
    tools = await toolset.get_tools(ctx)
    result = await toolset.call_tool('shout', {'value': 'hi'}, ctx, tools['shout'])
    assert result == 'HI'
    assert calls == ['hi']


async def test_call_tool_inside_context_is_checkpointed(absurd: AsyncAbsurd) -> None:
    calls: list[str] = []
    inner = _make_toolset(calls)
    toolset = AbsurdFunctionToolset(inner, step_name_prefix='a')
    absurd.register_task(name='shouter')(_noop)

    ctx = _run_context()
    # First attempt: the tool runs and its result is checkpointed.
    spawned_id: str
    async with running_task_context(absurd, 'shouter', max_attempts=2) as task_ctx:
        tools = await toolset.get_tools(ctx)
        first = await toolset.call_tool('shout', {'value': 'hi'}, ctx, tools['shout'])
        spawned_id = task_ctx.task_id

    # Second attempt: simulate a crash and retry. The tool must NOT run again; the
    # checkpointed result is served instead, so the side effect happens exactly once.
    async with reenter_running_task(absurd, spawned_id):
        replay = await toolset.call_tool('shout', {'value': 'hi'}, ctx, tools['shout'])

    assert first == replay == 'HI'
    assert calls == ['hi']  # the side effect ran exactly once across the crash
