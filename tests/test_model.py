from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

import pytest
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from pydantic_ai import ModelMessage, ModelResponse
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, TextPart
from pydantic_ai.models import CompletedStreamedResponse, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_absurd import AbsurdModel

from .conftest import running_task_context

pytestmark = pytest.mark.anyio


def _make_model(counter: dict[str, int]) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        counter['calls'] += 1
        return ModelResponse(parts=[TextPart(content=f'hello {counter["calls"]}')])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        counter['calls'] += 1
        yield f'hello {counter["calls"]}'

    return FunctionModel(fn, stream_function=stream_fn, model_name='test')


async def _noop_handler(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:  # pragma: no cover - not driven
    return None


def _register_noop(absurd: AsyncAbsurd) -> None:
    # Absurd's register_task decorator is typed for sync handlers; the async runtime
    # overload isn't reflected in types. See _agent.py for the same workaround.
    absurd.register_task(name='noop')(_noop_handler)


async def test_request_without_context_calls_model() -> None:
    counter = {'calls': 0}
    model = AbsurdModel(_make_model(counter), step_name_prefix='agent')
    resp = await model.request([], None, ModelRequestParameters())
    assert isinstance(resp, ModelResponse)
    assert counter['calls'] == 1


async def test_reserved_default_model_id_raises() -> None:
    counter = {'calls': 0}
    with pytest.raises(UserError, match="'default' is reserved"):
        AbsurdModel(_make_model(counter), step_name_prefix='agent', models={'default': _make_model(counter)})


async def test_for_run_default_returns_self() -> None:
    counter = {'calls': 0}
    model = AbsurdModel(_make_model(counter), step_name_prefix='agent', models={'cheap': _make_model(counter)})
    assert model.for_run(None) is model
    assert model.for_run('default') is model


async def test_for_run_registered_id_scopes_step_name() -> None:
    counter = {'calls': 0}
    cheap = _make_model(counter)
    model = AbsurdModel(_make_model(counter), step_name_prefix='agent', models={'cheap': cheap})
    bound = model.for_run('cheap')
    assert bound is not model
    assert bound.wrapped is cheap
    assert bound.request_step_name == 'agent__model.request.cheap'
    assert model.request_step_name == 'agent__model.request'


async def test_for_run_unregistered_raises() -> None:
    counter = {'calls': 0}
    model = AbsurdModel(_make_model(counter), step_name_prefix='agent')
    with pytest.raises(UserError, match="Model 'nope' is not registered"):
        model.for_run('nope')
    with pytest.raises(UserError, match='Arbitrary model instances cannot be used'):
        model.for_run(_make_model(counter))


async def test_request_stream_without_context_yields_raw_stream() -> None:
    counter = {'calls': 0}
    model = AbsurdModel(_make_model(counter), step_name_prefix='agent')
    async with model.request_stream([], None, ModelRequestParameters()) as stream:
        assert isinstance(stream, StreamedResponse)
        async for _ in stream:
            pass


async def test_request_inside_task_is_checkpointed(absurd: AsyncAbsurd) -> None:
    counter = {'calls': 0}
    model = AbsurdModel(_make_model(counter), step_name_prefix='agent')
    _register_noop(absurd)

    async with running_task_context(absurd, 'noop'):
        first = await model.request([], None, ModelRequestParameters())

    assert counter['calls'] == 1
    assert isinstance(first, ModelResponse)
    assert first.parts[0].content == 'hello 1'  # type: ignore[union-attr]


async def test_request_stream_inside_task_uses_completed_stream(absurd: AsyncAbsurd) -> None:
    counter = {'calls': 0}
    model = AbsurdModel(_make_model(counter), step_name_prefix='s')
    _register_noop(absurd)

    async with running_task_context(absurd, 'noop'):
        async with model.request_stream([], None, ModelRequestParameters()) as stream:
            assert isinstance(stream, CompletedStreamedResponse)
            first = stream.get().parts[0]
            assert isinstance(first, TextPart)
            assert first.content == 'hello 1'


async def test_request_stream_raises_if_event_handler_has_no_run_context(absurd: AsyncAbsurd) -> None:
    counter = {'calls': 0}

    async def handler(run_ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:  # pragma: no cover
        raise AssertionError('should not run')

    model = AbsurdModel(_make_model(counter), step_name_prefix='s', event_stream_handler=handler)
    _register_noop(absurd)

    async with running_task_context(absurd, 'noop'):
        with pytest.raises(RuntimeError, match='event_stream_handler'):
            async with model.request_stream([], None, ModelRequestParameters()):
                pass  # pragma: no cover


class _CountingStreamHandler:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, run_ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        self.calls += 1


async def test_request_stream_invokes_event_stream_handler(absurd: AsyncAbsurd) -> None:
    counter = {'calls': 0}
    handler = _CountingStreamHandler()
    model = AbsurdModel(_make_model(counter), step_name_prefix='h', event_stream_handler=handler)
    _register_noop(absurd)

    async with running_task_context(absurd, 'noop'):
        run_context = RunContext[None](deps=None, model=model, usage=RunUsage(), prompt='hi', messages=[])
        async with model.request_stream([], None, ModelRequestParameters(), run_context=run_context) as stream:
            assert isinstance(stream, CompletedStreamedResponse)
    assert handler.calls == 1
