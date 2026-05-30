from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from absurd_sdk import JsonValue
from pydantic import TypeAdapter
from pydantic_ai import ToolsetTool, WrapperToolset
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import InstructionPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from typing_extensions import Self

from ._utils import current_async_context

R = TypeVar('R')

Instructions = str | InstructionPart | Sequence[str | InstructionPart] | None

_tool_defs_map_adapter: TypeAdapter[dict[str, ToolDefinition]] = TypeAdapter(dict[str, ToolDefinition])
_instructions_adapter: TypeAdapter[Instructions] = TypeAdapter(Instructions)


def _serialize_tool_defs(defs: dict[str, ToolDefinition]) -> dict[str, JsonValue]:
    dumped = _tool_defs_map_adapter.dump_python(defs, mode='json')
    assert isinstance(dumped, dict)
    return dumped


def _deserialize_tool_defs(payload: dict[str, JsonValue]) -> dict[str, ToolDefinition]:
    return _tool_defs_map_adapter.validate_python(payload)


def _serialize_instructions(value: Instructions) -> JsonValue:
    dumped: JsonValue = _instructions_adapter.dump_python(value, mode='json')
    return dumped


def _deserialize_instructions(payload: JsonValue) -> Instructions:
    return _instructions_adapter.validate_python(payload)


class AbsurdMCPToolset(WrapperToolset[AgentDepsT]):
    """Durable wrapper around a Pydantic AI `MCPToolset`.

    Every `get_tools`, `get_instructions`, and `call_tool` is checkpointed into an
    Absurd step, so on replay the cached result is returned instead of re-hitting the
    MCP server. Tool definitions are also cached across steps to avoid redundant
    round-trips; the cache honors the wrapped toolset's `cache_tools` setting - set it
    to `False` when a server emits `tools/list_changed` notifications mid-workflow.
    """

    def __init__(
        self,
        wrapped: MCPToolset[AgentDepsT],
        *,
        step_name_prefix: str,
    ) -> None:
        super().__init__(wrapped)
        self._step_name_prefix = step_name_prefix
        id_suffix = f'__{wrapped.id}' if wrapped.id else ''
        self._name = f'{step_name_prefix}__mcp_server{id_suffix}'
        self._cached_tool_defs: dict[str, ToolDefinition] | None = None

    @property
    def _server(self) -> MCPToolset[AgentDepsT]:
        assert isinstance(self.wrapped, MCPToolset)
        return self.wrapped

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return self._server.tool_for_tool_def(tool_def)

    @property
    def id(self) -> str | None:
        return self.wrapped.id

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    def visit_and_replace(self, visitor: Callable[[Any], Any]) -> Any:
        return self

    async def _run_step(self, name_suffix: str, fn: Callable[[], Awaitable[R]]) -> R:
        task_ctx = current_async_context()
        if task_ctx is None:
            return await fn()
        return await task_ctx.step(f'{self._name}.{name_suffix}', fn)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if self._server.cache_tools and self._cached_tool_defs is not None:
            return {name: self.tool_for_tool_def(td) for name, td in self._cached_tool_defs.items()}

        async def _inner() -> dict[str, JsonValue]:
            tools = await super(AbsurdMCPToolset, self).get_tools(ctx)
            return _serialize_tool_defs({name: tool.tool_def for name, tool in tools.items()})

        if current_async_context() is None:
            return await super().get_tools(ctx)

        payload = await self._run_step('get_tools', _inner)
        tool_defs = _deserialize_tool_defs(payload)
        result = {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.items()}
        if self._server.cache_tools:
            self._cached_tool_defs = tool_defs
        return result

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> Instructions:
        result = await super().get_instructions(ctx)
        if result is not None:  # pragma: no cover - fast path when the wrapped server is already entered
            return result

        if current_async_context() is None:
            return None

        if not self._server.include_instructions:
            return None

        async def _inner() -> JsonValue:
            async with self.wrapped:
                return _serialize_instructions(await super(AbsurdMCPToolset, self).get_instructions(ctx))

        payload = await self._run_step('get_instructions', _inner)
        return _deserialize_instructions(payload)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        async def _inner() -> Any:
            return await self._server.call_tool(name, tool_args, ctx, tool)

        return await self._run_step('call_tool', _inner)
