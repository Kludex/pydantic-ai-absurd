from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from absurd_sdk import JsonValue
from pydantic import TypeAdapter
from pydantic_ai import ModelMessage, ModelResponse, models
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from ._utils import current_async_context

_response_adapter: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)


def _serialize(response: ModelResponse) -> dict[str, JsonValue]:
    dumped = _response_adapter.dump_python(response, mode='json')
    assert isinstance(dumped, dict)
    return dumped


def _deserialize(payload: dict[str, JsonValue]) -> ModelResponse:
    return _response_adapter.validate_python(payload)


class AbsurdModel(WrapperModel):
    """A `WrapperModel` that checkpoints `request()` and `request_stream()` via Absurd.

    When the wrapping agent is executed inside an Absurd task context, each call is
    wrapped in `ctx.step()` - so after a crash the same checkpoint returns the cached
    `ModelResponse` on replay and no new tokens are spent.

    `ModelResponse` is serialized to JSON for storage via Pydantic (Absurd only stores
    JSON values in checkpoints); on replay, the stored JSON is round-tripped back into
    a `ModelResponse`.
    """

    def __init__(
        self,
        model: Model,
        *,
        step_name_prefix: str,
        event_stream_handler: EventStreamHandler[Any] | None = None,
        models: Mapping[str, Model] | None = None,
        _model_id: str | None = None,
        _models_by_id: dict[str, Model] | None = None,
    ) -> None:
        super().__init__(model)
        self._step_name_prefix = step_name_prefix
        self.event_stream_handler = event_stream_handler
        self._model_id = _model_id

        if _models_by_id is not None:
            self._models_by_id = _models_by_id
            return

        self._models_by_id = {'default': model}
        for model_id, instance in (models or {}).items():
            if model_id == 'default':
                raise UserError("Model ID 'default' is reserved for the agent's primary model.")
            self._models_by_id[model_id] = instance

    @property
    def request_step_name(self) -> str:
        return self._step_name('request')

    @property
    def request_stream_step_name(self) -> str:
        return self._step_name('request_stream')

    def _step_name(self, op: str) -> str:
        suffix = f'.{self._model_id}' if self._model_id is not None else ''
        return f'{self._step_name_prefix}__model.{op}{suffix}'

    def _get_model_id(self, model: models.Model | models.KnownModelName | str | None) -> str | None:
        if model in (None, 'default'):
            return None
        if isinstance(model, Model):
            model_id = next((mid for mid, m in self._models_by_id.items() if m is model), None)
            if model_id is None:
                raise UserError(
                    'Arbitrary model instances cannot be used at run time with Absurd; '
                    'register the model via `models=` or reference a registered model by id.'
                )
            return None if model_id == 'default' else model_id
        if model not in self._models_by_id:
            raise UserError(
                f'Model {model!r} is not registered; pass it to `AbsurdAgent(..., models=...)` '
                'or use a registered model id.'
            )
        return model

    def resolve_model(self, model: models.Model | models.KnownModelName | str | None) -> Model:
        model_id = self._get_model_id(model)
        return self.wrapped if model_id is None else self._models_by_id[model_id]

    def for_run(self, model: models.Model | models.KnownModelName | str | None) -> AbsurdModel:
        model_id = self._get_model_id(model)
        if model_id is None:
            return self
        return AbsurdModel(
            self._models_by_id[model_id],
            step_name_prefix=self._step_name_prefix,
            event_stream_handler=self.event_stream_handler,
            _model_id=model_id,
            _models_by_id=self._models_by_id,
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        ctx = current_async_context()
        if ctx is None:
            return await super().request(messages, model_settings, model_request_parameters)

        async def _inner() -> dict[str, JsonValue]:
            response = await super(AbsurdModel, self).request(messages, model_settings, model_request_parameters)
            return _serialize(response)

        payload = await ctx.step(self.request_step_name, _inner)
        return _deserialize(payload)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        ctx = current_async_context()
        if ctx is None:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                yield stream
                return

        async def _inner() -> dict[str, JsonValue]:
            async with super(AbsurdModel, self).request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed:
                if self.event_stream_handler is not None:
                    if run_context is None:
                        raise RuntimeError(
                            'AbsurdModel cannot stream without a `run_context`; '
                            'set an `event_stream_handler` on the agent and use `agent.run()`.'
                        )
                    # EventStreamHandler expects AsyncIterable[AgentStreamEvent], but at the
                    # model layer only ModelResponseStreamEvents are available. Mirrors the
                    # pattern used by pydantic-ai's own DBOS/Prefect/Temporal adapters.
                    await self.event_stream_handler(run_context, streamed)

                async for _ in streamed:
                    pass
            return _serialize(streamed.get())

        payload = await ctx.step(self.request_stream_step_name, _inner)
        yield CompletedStreamedResponse(_deserialize(payload), model_request_parameters=model_request_parameters)
