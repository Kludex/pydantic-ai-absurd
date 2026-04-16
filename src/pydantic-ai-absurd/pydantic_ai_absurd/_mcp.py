from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from absurd_sdk import JsonValue
from pydantic import TypeAdapter
from pydantic_ai import AbstractToolset, ToolsetTool, WrapperToolset
from pydantic_ai.mcp import MCPServer
from pydantic_ai.messages import InstructionPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from typing_extensions import Self

from ._utils import StepConfig, current_async_context

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


class AbsurdMCPToolset(WrapperToolset[AgentDepsT], ABC):
    """Base for MCP toolsets whose calls are checkpointed into Absurd steps.

    `call_tool` lives on the concrete subclass (`AbsurdMCPServer`, `AbsurdFastMCPToolset`)
    so it can declare the actual return type of the wrapped toolset (`ToolResult` for
    MCPServer). This base provides the step dispatch helpers only.
    """

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        step_name_prefix: str,
        step_config: StepConfig | None = None,
    ) -> None:
        super().__init__(wrapped)
        self._step_config: StepConfig = step_config or {}
        self._step_name_prefix = step_name_prefix
        id_suffix = f'__{wrapped.id}' if wrapped.id else ''
        self._name = f'{step_name_prefix}__mcp_server{id_suffix}'

    @abstractmethod
    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        raise NotImplementedError

    @property
    def id(self) -> str | None:
        return self.wrapped.id

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return self

    async def _run_step(self, name_suffix: str, fn: Callable[[], Awaitable[R]]) -> R:
        task_ctx = current_async_context()
        if task_ctx is None:
            return await fn()
        return await task_ctx.step(f'{self._name}.{name_suffix}', fn)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        async def _inner() -> dict[str, JsonValue]:
            tools = await super(AbsurdMCPToolset, self).get_tools(ctx)
            return _serialize_tool_defs({name: tool.tool_def for name, tool in tools.items()})

        if current_async_context() is None:
            return await super().get_tools(ctx)

        payload = await self._run_step('get_tools', _inner)
        tool_defs = _deserialize_tool_defs(payload)
        return {name: self.tool_for_tool_def(tool_def) for name, tool_def in tool_defs.items()}

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        result = await super().get_instructions(ctx)
        if result is not None:  # pragma: no cover - fast path when the wrapped server is already entered
            return result

        if current_async_context() is None:
            return None

        if not isinstance(self.wrapped, (MCPServer, FastMCPToolset)):  # pragma: no cover - defensive
            return None
        if not self.wrapped.include_instructions:
            return None

        async def _inner() -> JsonValue:
            async with self.wrapped:
                return _serialize_instructions(await super(AbsurdMCPToolset, self).get_instructions(ctx))

        payload = await self._run_step('get_instructions', _inner)
        return _deserialize_instructions(payload)
