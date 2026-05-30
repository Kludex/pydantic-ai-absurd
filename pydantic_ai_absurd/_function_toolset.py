from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic_ai import FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.tools import AgentDepsT, RunContext
from typing_extensions import Self

from ._utils import current_async_context

R = TypeVar('R')


class AbsurdFunctionToolset(WrapperToolset[AgentDepsT]):
    """Durable wrapper around a Pydantic AI `FunctionToolset`.

    Every `call_tool` is checkpointed into an Absurd step, so on replay the cached
    return value is returned instead of running the tool again. This makes a tool with
    a side effect (sending an email, charging a card, writing a row) run exactly once
    across crashes, the same guarantee model and MCP calls get.

    The tool's return value is stored in Postgres, so it must be JSON-serializable.
    Listing the tools is not checkpointed: a `FunctionToolset` is local Python, so
    `get_tools` is cheap and deterministic and there's nothing to cache.
    """

    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
        *,
        step_name_prefix: str,
    ) -> None:
        super().__init__(wrapped)
        self._step_name_prefix = step_name_prefix
        id_suffix = f'__{wrapped.id}' if wrapped.id else ''
        self._name = f'{step_name_prefix}__function_toolset{id_suffix}'

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

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        async def _inner() -> Any:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)

        return await self._run_step(f'call_tool:{name}', _inner)
