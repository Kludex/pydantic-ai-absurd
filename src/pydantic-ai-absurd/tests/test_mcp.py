from __future__ import annotations

import sys
from pathlib import Path

import pytest
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from fastmcp import FastMCP
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_absurd import AbsurdFastMCPToolset, AbsurdMCPServer

from .conftest import running_task_context

MCP_SCRIPT = str(Path(__file__).resolve().parent / 'fixtures' / 'mcp_server.py')


async def _return_hello() -> str:
    return 'hello'


pytestmark = pytest.mark.anyio


async def _noop(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:  # pragma: no cover
    return None


def _make_server() -> FastMCPToolset[None]:
    server: FastMCP[None] = FastMCP(name='calc')

    @server.tool
    def add(a: int, b: int) -> int:
        return a + b

    return FastMCPToolset(server)


def _run_context() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt='x', messages=[])


async def test_fastmcp_server_access(absurd: AsyncAbsurd) -> None:
    toolset = AbsurdFastMCPToolset(_make_server(), step_name_prefix='a')
    assert toolset._server is toolset.wrapped


async def test_fastmcp_visit_and_replace_returns_self(absurd: AsyncAbsurd) -> None:
    toolset = AbsurdFastMCPToolset(_make_server(), step_name_prefix='a')

    def visitor(t: FastMCPToolset[None]) -> FastMCPToolset[None]:  # pragma: no cover
        return t

    # visit_and_replace on the Absurd wrapper returns itself (already wrapped).
    assert toolset.visit_and_replace(visitor) is toolset  # type: ignore[arg-type]


async def test_fastmcp_id_passthrough(absurd: AsyncAbsurd) -> None:
    inner = _make_server()
    toolset = AbsurdFastMCPToolset(inner, step_name_prefix='a')
    assert toolset.id == inner.id


async def test_fastmcp_aenter_aexit(absurd: AsyncAbsurd) -> None:
    toolset = AbsurdFastMCPToolset(_make_server(), step_name_prefix='a')
    async with toolset as entered:
        assert entered is toolset


async def test_fastmcp_get_tools_and_call_tool_are_checkpointed(absurd: AsyncAbsurd) -> None:
    toolset = AbsurdFastMCPToolset(_make_server(), step_name_prefix='a')
    absurd.register_task(name='noop')(_noop)  # type: ignore[arg-type]

    run_context = _run_context()
    async with running_task_context(absurd, 'noop'):
        tools = await toolset.get_tools(run_context)
        assert 'add' in tools
        result = await toolset.call_tool('add', {'a': 2, 'b': 3}, run_context, tools['add'])
    assert result


async def test_fastmcp_get_tools_without_context_passes_through(absurd: AsyncAbsurd) -> None:
    toolset = AbsurdFastMCPToolset(_make_server(), step_name_prefix='a')
    tools = await toolset.get_tools(_run_context())
    assert 'add' in tools


async def test_fastmcp_get_instructions_without_context_passes_through(absurd: AsyncAbsurd) -> None:
    toolset = AbsurdFastMCPToolset(_make_server(), step_name_prefix='a')
    result = await toolset.get_instructions(_run_context())
    # FastMCP servers without include_instructions return None
    assert result is None


async def test_fastmcp_get_instructions_inside_context_returns_none_when_disabled(absurd: AsyncAbsurd) -> None:
    toolset = AbsurdFastMCPToolset(_make_server(), step_name_prefix='a')
    absurd.register_task(name='noop')(_noop)  # type: ignore[arg-type]

    async with running_task_context(absurd, 'noop'):
        result = await toolset.get_instructions(_run_context())
    assert result is None


async def test_fastmcp_get_instructions_inside_context_with_include(absurd: AsyncAbsurd) -> None:
    # Enable include_instructions on the FastMCP server so the step dispatch path runs.
    server: FastMCP[None] = FastMCP(name='hello', instructions='Be brief.')
    inner = FastMCPToolset[None](server, include_instructions=True)
    toolset = AbsurdFastMCPToolset(inner, step_name_prefix='a')
    absurd.register_task(name='noop')(_noop)  # type: ignore[arg-type]

    async with running_task_context(absurd, 'noop'):
        result = await toolset.get_instructions(_run_context())
    assert result is not None


async def test_run_step_without_context_passes_through(absurd: AsyncAbsurd) -> None:
    toolset = AbsurdFastMCPToolset(_make_server(), step_name_prefix='a')
    assert await toolset._run_step('x', _return_hello) == 'hello'


def _make_mcp_server() -> MCPServerStdio:
    return MCPServerStdio(command=sys.executable, args=[MCP_SCRIPT])


async def test_mcp_server_get_tools_and_call_tool_inside_context(absurd: AsyncAbsurd) -> None:
    server = _make_mcp_server()
    toolset = AbsurdMCPServer(server, step_name_prefix='a')
    absurd.register_task(name='noop')(_noop)  # type: ignore[arg-type]

    run_context = _run_context()
    async with running_task_context(absurd, 'noop'):
        async with server:
            tools = await toolset.get_tools(run_context)
            assert 'add' in tools
            # Second call hits the cache path (cache_tools defaults to True).
            cached = await toolset.get_tools(run_context)
            assert cached.keys() == tools.keys()
            result = await toolset.call_tool('add', {'a': 4, 'b': 5}, run_context, tools['add'])
    assert result is not None


async def test_mcp_server_get_tools_without_cache(absurd: AsyncAbsurd) -> None:
    server = MCPServerStdio(command=sys.executable, args=[MCP_SCRIPT], cache_tools=False)
    toolset = AbsurdMCPServer(server, step_name_prefix='a')
    absurd.register_task(name='noop')(_noop)  # type: ignore[arg-type]

    run_context = _run_context()
    async with running_task_context(absurd, 'noop'):
        async with server:
            first = await toolset.get_tools(run_context)
            second = await toolset.get_tools(run_context)
    assert first.keys() == second.keys() == {'add'}
