from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

from absurd_sdk import JsonValue
from pydantic import TypeAdapter
from pydantic_ai import FunctionToolset
from pydantic_ai.agent import EventStreamHandler, ParallelExecutionMode
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import WrapModelRequestHandler, WrapRunHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind, reject_unsupported_runtime_toolsets
from pydantic_ai.durable_exec._utils import DurableModel, StreamedActivityResult, capture_event_stream
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from ._function_toolset import AbsurdFunctionToolset
from ._mcp import AbsurdMCPToolset
from ._model import _deserialize, _serialize
from ._utils import current_async_context

_events_adapter: TypeAdapter[list[ModelResponseStreamEvent]] = TypeAdapter(list[ModelResponseStreamEvent])

AbsurdParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']
"""Tool-call execution modes usable with Absurd. A subset of `ParallelExecutionMode`: `'parallel'`
is excluded because Absurd disambiguates repeated step names with an encounter-order counter, so
checkpoints must be reached in a deterministic order for a replay to line up with them."""


@dataclass(init=False)
class AbsurdDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by checkpointing I/O into Absurd steps.

    Attach it to an agent via `capabilities=[AbsurdDurability()]` and call `agent.run()`
    inside an Absurd task handler: every model request, MCP call, and function tool call
    is wrapped in `ctx.step(...)`, so a worker crash mid-run resumes from the last
    completed step instead of restarting — no tokens are re-spent, and side effects run
    once. Outside a task the capability is transparent and the run is a normal,
    non-durable agent run.

    The capability discovers the agent's model, name, and toolsets automatically when it
    is bound to the agent. Step results are stored in Postgres as JSON, so a checkpointed
    tool's return value must be JSON-serializable.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_absurd import AbsurdDurability

        agent = Agent('openai:gpt-5.2', name='analyst', capabilities=[AbsurdDurability()])

        @absurd.register_task(name='analyse')
        async def analyse(params, ctx):
            result = await agent.run(params['prompt'])
            return {'output': result.output}
        ```
    """

    engine_name = 'Absurd'
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset(
        {'function', 'mcp', 'dynamic'}
    )

    _durable_unit_noun = 'step'
    _durable_container_noun = 'task'

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        parallel_execution_mode: AbsurdParallelExecutionMode = 'sequential',
    ) -> None:
        """Create an AbsurdDurability capability.

        The agent's model, name, and toolsets are discovered automatically.

        Args:
            models: Optional additional models keyed by ID for runtime model switching via
                `agent.run(model='<id>')`. The agent's primary model is always registered as
                `'default'`; the ID is folded into the checkpoint step name so a replay
                resolves to the same model.
            event_stream_handler: Optional event stream handler. Model events are handled
                live inside the model-request step; each tool event is handled in its own
                checkpointed step.
            name: Unique agent name used as the prefix for every checkpoint step. Defaults
                to the agent's `name` when the capability is bound.
            parallel_execution_mode: Tool-call execution mode applied for the duration of
                every run. Defaults to `'sequential'`. `'parallel'` is excluded by type:
                Absurd disambiguates repeated step names with a counter, so steps must be
                reached in a deterministic order for a replay to line up with its
                checkpoints.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self._parallel_execution_mode = cast(ParallelExecutionMode, parallel_execution_mode)
        self._wrappers_by_leaf: dict[int, WrapperToolset[AgentDepsT]] = {}
        self._construction_leaves: set[int] = set()
        self._default_model_id: str | None = None

    @property
    def in_durable_context(self) -> bool:
        return current_async_context() is not None

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        # Absurd steps are ad-hoc `ctx.step(...)` calls, so unlike Temporal there is nothing
        # to register up front beyond the durable toolset wrappers. Wrappers are keyed by leaf
        # *instance* rather than toolset `id`: it keeps id-less toolsets working (with the same
        # step names the deprecated wrapper used), and it stops a runtime toolset that happens
        # to share an `id` with a construction-time one (e.g. `override(tools=...)` recreating
        # the agent's own toolset) from being silently swapped for the registered wrapper.
        self._default_model_id = agent.model if isinstance(agent.model, str) else None
        self._wrappers_by_leaf = {}
        self._construction_leaves = set()
        seen_ids: dict[str, AbstractToolset[AgentDepsT]] = {}

        def register(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            ts_id = ts.id
            if ts_id is not None:
                existing = seen_ids.get(ts_id)
                if existing is not None and existing is not ts:
                    raise UserError(
                        f'Two toolsets have the same `id` {ts_id!r}. Toolset `id`s must be unique among all '
                        f"toolsets registered with the same agent, as they identify the toolset's steps "
                        'within the task.'
                    )
                seen_ids[ts_id] = ts
            if id(ts) not in self._construction_leaves:
                self._construction_leaves.add(id(ts))
                wrapper = self._wrap_leaf_toolset(ts)
                if wrapper is not None:
                    self._wrappers_by_leaf[id(ts)] = wrapper
            return ts

        for toolset in agent.toolsets:
            toolset.visit_and_replace(register)

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        if isinstance(ts, MCPToolset):
            return AbsurdMCPToolset(wrapped=ts, step_name_prefix=self.name)
        if isinstance(ts, FunctionToolset):
            return AbsurdFunctionToolset(wrapped=ts, step_name_prefix=self.name)
        return None

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Swap construction-time leaves for their durable wrappers, rejecting runtime additions.

        A leaf that wasn't seen at binding time - added per-run via `run(toolsets=...)`, an
        `override(...)`, or another capability - has no durable wrapper, so executing it inside
        a task would bypass checkpointing and re-run its side effects on recovery. Outside a
        task everything passes through and the agent behaves like a regular agent.

        Candidates are collected with `visit_and_replace`, the same leaf-only walk registration
        used. `apply` would also visit wrapper nodes Pydantic AI inserts itself (e.g. the
        `CapabilityOwnedToolset` around a toolset contributed by `AbstractCapability.get_toolset()`),
        which are never registered as leaves and would otherwise be misreported as runtime
        toolsets - naming the inner toolset that *was* registered in the error.

        Sharing that one walk means the rejection is raised *after* the swap has built the new
        tree, rather than before it as it used to be. That is deliberate and safe: `swap` only
        reads the registration maps and appends to a local list, the swapped tree is a fresh
        structure that is discarded when the error propagates, and the raise still happens before
        this method returns, so no un-checkpointed toolset can reach the run.
        """
        in_durable_context = self.in_durable_context
        runtime_leaves: list[AbstractToolset[AgentDepsT]] = []

        def swap(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            if in_durable_context and id(ts) not in self._construction_leaves:
                runtime_leaves.append(ts)
            return self._wrappers_by_leaf.get(id(ts), ts)

        swapped = toolset.visit_and_replace(swap)
        reject_unsupported_runtime_toolsets(
            runtime_leaves,
            unsupported_kinds=self._unsupported_runtime_toolset_kinds,
            engine=self.engine_name,
        )
        return swapped

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        task_ctx = current_async_context()
        assert task_ctx is not None  # pragma: no cover - only dispatched inside a durable context
        handler = self._event_stream_handler
        assert handler is not None  # pragma: no cover - only dispatched when a handler is set

        async def _inner() -> None:
            await handler(ctx, self._single_event_stream(event))

        # Checkpoint the handler call so its side effects don't re-run on recovery.
        await task_ctx.step(f'{self.name}__event_stream_handler', _inner)

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Apply the configured parallel-execution mode for every entry point."""
        agent = self._agent
        if agent is None:  # pragma: no cover
            return await handler()
        with agent.parallel_tool_call_execution_mode(self._parallel_execution_mode):
            return await handler()

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Checkpoint model requests into Absurd steps when inside a task."""
        task_ctx = current_async_context()
        if task_ctx is None:
            return await handler(request_context)

        # The step runs in-process, so the model needs no cross-boundary rebuild; the
        # model id is folded into the step name so a replay maps each checkpoint back
        # to the model it was recorded for (and stays compatible with the deprecated
        # `AbsurdAgent` step names).
        model_id = self._model_id_for_request(ctx, request_context)
        if model_id is not None and model_id == self._default_model_id:
            # A string default stays raw through binding, so its requests carry the string
            # as provenance - but it's still the agent's default model, and the wrapper
            # (which resolved the default eagerly) checkpointed it without a suffix.
            model_id = None
        step_suffix = '' if model_id is None else f'.{model_id}'
        model = request_context.model

        async def request_segment(request: ModelRequestContext) -> ModelResponse:
            async def _inner() -> dict[str, JsonValue]:
                response = await request.model.request(
                    request.messages, request.model_settings, request.model_request_parameters
                )
                return _serialize(response)

            payload = await task_ctx.step(f'{self.name}__model.request{step_suffix}', _inner)
            return _deserialize(payload)

        async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
            async def _inner() -> dict[str, JsonValue]:
                async with request.model.request_stream(
                    request.messages, request.model_settings, request.model_request_parameters, ctx
                ) as streamed:
                    events = await capture_event_stream(
                        run_context=ctx, stream=streamed, handler=self._event_stream_handler
                    )
                return {
                    'response': _serialize(streamed.get()),
                    'events': _events_adapter.dump_python(events, mode='json'),
                }

            payload = await task_ctx.step(f'{self.name}__model.request_stream{step_suffix}', _inner)
            if 'response' not in payload:
                # Checkpoint written by the deprecated `AbsurdAgent` wrapper: a bare
                # `ModelResponse` with no captured events. Synthesize the events from the
                # response parts (as the wrapper's replay did) so a run started under the
                # wrapper resumes correctly under the capability.
                response = _deserialize(payload)
                completed = CompletedStreamedResponse(
                    response, model_request_parameters=request.model_request_parameters, events=True
                )
                return StreamedActivityResult(response=response, events=[event async for event in completed])
            response_payload = payload['response']
            assert isinstance(response_payload, dict)
            return StreamedActivityResult(
                response=_deserialize(response_payload),
                events=_events_adapter.validate_python(payload['events']),
            )

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            async def _inner() -> None:
                await model.cancel_suspended_response(response)

            await task_ctx.step(f'{self.name}__model.cancel_suspended_response{step_suffix}', _inner)

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        return await handler(request_context)
