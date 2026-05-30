from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any

from absurd_sdk import AsyncAbsurd
from pydantic_ai import (
    AbstractToolset,
    _instructions,
    _utils,
    messages as _messages,
    models,
    usage as _usage,
)
from pydantic_ai.agent import (
    AbstractAgent,
    AgentRun,
    AgentRunResult,
    EventStreamHandler,
    ParallelExecutionMode,
    WrapperAgent,
)
from pydantic_ai.agent.abstract import AgentMetadata, AgentModelSettings, AgentRetries, RunOutputDataT
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.tools import AgentDepsT, AgentNativeTool, DeferredToolResults, Tool, ToolFuncEither

from ._mcp import AbsurdMCPToolset
from ._model import AbsurdModel
from ._utils import StepConfig, current_async_context, require_async_context

if TYPE_CHECKING:
    from pydantic_ai.agent.spec import AgentSpec


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
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,  # type: ignore[assignment]
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AgentRunResult[Any]:
        self._reject_non_absurd_model(model)
        require_async_context()
        with self._absurd_overrides():
            return await super().run(
                user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                conversation_id=conversation_id,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                retries=retries,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                capabilities=capabilities,
                spec=spec,
            )

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,  # type: ignore[assignment]
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AgentRunResult[Any]:
        raise UserError(
            'AbsurdAgent.run_sync() is not supported: the Absurd task handler is already async. '
            'Use `await agent.run(...)` inside your task.'
        )

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,  # type: ignore[assignment]
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        self._reject_non_absurd_model(model)
        kwargs: dict[str, Any] = dict(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            conversation_id=conversation_id,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            retries=retries,
            infer_name=infer_name,
            toolsets=toolsets,
            capabilities=capabilities,
            spec=spec,
        )
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
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,  # type: ignore[assignment]
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]]:
        raise UserError(
            '`agent.run_stream_events()` cannot be used with Absurd. Set an '
            '`event_stream_handler` on the agent and use `agent.run()` instead.'
        )

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,  # type: ignore[assignment]
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
        if current_async_context() is not None:
            raise UserError(
                '`agent.run_stream()` cannot be used inside an Absurd task. Set an '
                '`event_stream_handler` on the agent and use `agent.run()` instead.'
            )
        async with super().run_stream(
            user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            conversation_id=conversation_id,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            retries=retries,
            infer_name=infer_name,
            toolsets=toolsets,
            event_stream_handler=event_stream_handler,
            capabilities=capabilities,
            spec=spec,
        ) as result:
            yield result

    @staticmethod
    def _reject_non_absurd_model(model: object) -> None:
        if model is not None and not isinstance(model, AbsurdModel):
            raise UserError('Non-Absurd model cannot be overridden at run time; set `model` at agent construction.')

    @contextmanager
    def override(
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        native_tools: Sequence[AgentNativeTool[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        instructions: _instructions.AgentInstructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
        model_settings: AgentModelSettings[AgentDepsT] | _utils.Unset = _utils.UNSET,
        retries: int | AgentRetries | _utils.Unset = _utils.UNSET,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> Iterator[None]:
        if _utils.is_set(model) and not isinstance(model, AbsurdModel):
            raise UserError('Non-Absurd model cannot be overridden inside an Absurd task; set it at construction time.')
        with super().override(
            name=name,
            deps=deps,
            model=model,
            toolsets=toolsets,
            tools=tools,
            native_tools=native_tools,
            instructions=instructions,
            model_settings=model_settings,
            retries=retries,
            spec=spec,
        ):
            yield
