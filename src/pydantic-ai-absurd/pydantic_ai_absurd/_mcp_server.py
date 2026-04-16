from __future__ import annotations

from typing import Any

from pydantic_ai import ToolsetTool
from pydantic_ai.mcp import MCPServer, ToolResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from ._mcp import AbsurdMCPToolset
from ._utils import StepConfig


class AbsurdMCPServer(AbsurdMCPToolset[AgentDepsT]):
    """Durable wrapper around a Pydantic AI `MCPServer`.

    Tool definitions are cached across steps to avoid redundant MCP round-trips. The
    cache honors the wrapped server's `cache_tools` setting - set it to `False` when a
    server emits `tools/list_changed` notifications mid-workflow.
    """

    def __init__(
        self,
        wrapped: MCPServer,
        *,
        step_name_prefix: str,
        step_config: StepConfig | None = None,
    ) -> None:
        super().__init__(wrapped, step_name_prefix=step_name_prefix, step_config=step_config)
        self._cached_tool_defs: dict[str, ToolDefinition] | None = None

    @property
    def _server(self) -> MCPServer:
        assert isinstance(self.wrapped, MCPServer)
        return self.wrapped

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[AgentDepsT]:
        return self._server.tool_for_tool_def(tool_def)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if self._server.cache_tools and self._cached_tool_defs is not None:
            return {name: self.tool_for_tool_def(td) for name, td in self._cached_tool_defs.items()}

        result = await super().get_tools(ctx)
        if self._server.cache_tools:
            self._cached_tool_defs = {name: tool.tool_def for name, tool in result.items()}
        return result

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> ToolResult:
        async def _inner() -> ToolResult:
            return await self._server.call_tool(name, tool_args, ctx, tool)

        return await self._run_step('call_tool', _inner)
