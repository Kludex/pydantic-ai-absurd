from __future__ import annotations

from typing import Any

from pydantic_ai import ToolsetTool
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

from ._mcp import AbsurdMCPToolset
from ._utils import StepConfig


class AbsurdFastMCPToolset(AbsurdMCPToolset[AgentDepsT]):
    """Durable wrapper around a Pydantic AI `FastMCPToolset`."""

    def __init__(
        self,
        wrapped: FastMCPToolset[AgentDepsT],
        *,
        step_name_prefix: str,
        step_config: StepConfig | None = None,
    ) -> None:
        super().__init__(wrapped, step_name_prefix=step_name_prefix, step_config=step_config)

    @property
    def _server(self) -> FastMCPToolset[AgentDepsT]:
        assert isinstance(self.wrapped, FastMCPToolset)
        return self.wrapped

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return self._server.tool_for_tool_def(tool_def)

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
