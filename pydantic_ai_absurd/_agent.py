from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from absurd_sdk import AsyncAbsurd
from pydantic_ai import (
    AbstractToolset,
    _utils,
    messages as _messages,
)
from pydantic_ai.agent import (
    AbstractAgent,
    AgentRun,
    AgentRunResult,
    EventStreamHandler,
    ParallelExecutionMode,
    WrapperAgent,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.tools import AgentDepsT

from ._mcp import AbsurdMCPToolset
from ._model import AbsurdModel
from ._utils import StepConfig, current_async_context, require_async_context


class AbsurdAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    """Wrap a Pydantic AI agent so its `run()` is durable when called inside an Absurd task.

    Call `await agent.run(...)` from within an Absurd task handler and every model call and
    MCP call inside the run is checkpointed via `ctx.step(...)`. A worker crash mid-run
    re-runs the task, but the checkpointed steps return their cached results - so the run
    resumes from the last completed step instead of restarting, and no tokens are re-spent.

    The wrapped model is replaced with `AbsurdModel`; any `MCPToolset` is replaced with
    its Absurd counterpart. Plain function toolsets are left untouched - their Python
    side-effects are expected to be idempotent and cheap to re-run (the expensive,
    non-idempotent things live behind LLM calls and MCP calls, both of which are
    checkpointed).

    You author the task; the agent is a durable callable inside it:

        agent = AbsurdAgent(Agent('openai:gpt-5.2', name='analyst'), absurd, name='analyst')

        @absurd.register_task(name='analyse')
        async def analyse(params, ctx):
            result = await agent.run(params['prompt'])
            return result.output
    """

    _parallel_execution_mode: ParallelExecutionMode

    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        absurd: AsyncAbsurd,
        *,
        name: str | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        model_step_config: StepConfig | None = None,
        mcp_step_config: StepConfig | None = None,
        parallel_execution_mode: ParallelExecutionMode = 'sequential',
    ) -> None:
        super().__init__(wrapped)

        self._absurd = absurd
        self._name = name or wrapped.name
        self._event_stream_handler = event_stream_handler
        self._parallel_execution_mode = parallel_execution_mode
        if self._name is None:
            raise UserError(
                'An agent needs a unique `name` to be used with Absurd. The name is '
                'used as the prefix for every checkpoint step.'
            )

        self._model_step_config: StepConfig = model_step_config or {}
        self._mcp_step_config: StepConfig = mcp_step_config or {}

        if not isinstance(wrapped.model, Model):
            raise UserError('An agent needs a `model` set at construction time to be wrapped with AbsurdAgent.')

        self._model = AbsurdModel(
            wrapped.model,
            step_name_prefix=self._name,
            step_config=self._model_step_config,
            event_stream_handler=self.event_stream_handler,
        )

        self._toolsets: Sequence[AbstractToolset[AgentDepsT]] = [
            toolset.visit_and_replace(self._absurdify_toolset) for toolset in wrapped.toolsets
        ]

    def _absurdify_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        if isinstance(toolset, MCPToolset):
            return AbsurdMCPToolset(
                wrapped=toolset,
                step_name_prefix=self._step_prefix,
                step_config=self._mcp_step_config,
            )
        return toolset

    @property
    def _step_prefix(self) -> str:
        assert self._name is not None  # pragma: no cover - enforced in __init__
        return self._name

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:  # pragma: no cover
        raise UserError('The agent name cannot be changed after creation; create a new AbsurdAgent instead.')

    @property
    def model(self) -> Model:
        return self._model

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        return self._event_stream_handler or super().event_stream_handler

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        with self._absurd_overrides():
            return super().toolsets

    @contextmanager
    def _absurd_overrides(self) -> Iterator[None]:
        with (
            super().override(model=self._model, toolsets=self._toolsets, tools=[]),
            self.parallel_tool_call_execution_mode(self._parallel_execution_mode),
        ):
            yield

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[Any]:
        self._reject_non_absurd_model(kwargs.get('model'))
        require_async_context()
        with self._absurd_overrides():
            result: AgentRunResult[Any] = await super().run(user_prompt, **kwargs)
        return result

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        **kwargs: Any,
    ) -> AgentRunResult[Any]:
        raise UserError(
            'AbsurdAgent.run_sync() is not supported: the Absurd task handler is already async. '
            'Use `await agent.run(...)` inside your task.'
        )

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        self._reject_non_absurd_model(kwargs.get('model'))
        if current_async_context() is None:
            async with super().iter(user_prompt, **kwargs) as run:
                yield run
                return
        with self._absurd_overrides():
            async with super().iter(user_prompt, **kwargs) as run:
                yield run

    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        **kwargs: Any,
    ) -> Any:
        raise UserError(
            '`agent.run_stream_events()` cannot be used with Absurd. Set an '
            '`event_stream_handler` on the agent and use `agent.run()` instead.'
        )

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
        if current_async_context() is not None:
            raise UserError(
                '`agent.run_stream()` cannot be used inside an Absurd task. Set an '
                '`event_stream_handler` on the agent and use `agent.run()` instead.'
            )
        async with super().run_stream(user_prompt, **kwargs) as result:
            yield result

    @staticmethod
    def _reject_non_absurd_model(model: object) -> None:
        if model is not None and not isinstance(model, AbsurdModel):
            raise UserError('Non-Absurd model cannot be overridden at run time; set `model` at agent construction.')

    @contextmanager
    def override(self, **kwargs: Any) -> Iterator[None]:
        model = kwargs.get('model', _utils.UNSET)
        if _utils.is_set(model) and not isinstance(model, AbsurdModel):
            raise UserError('Non-Absurd model cannot be overridden inside an Absurd task; set it at construction time.')
        with super().override(**kwargs):
            yield
