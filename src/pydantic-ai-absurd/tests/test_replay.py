from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from pydantic_ai import ModelMessage, ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_absurd import AbsurdModel

from .conftest import reenter_running_task, running_task_context

pytestmark = pytest.mark.anyio


async def test_checkpointed_response_survives_crash_and_is_not_reissued(absurd: AsyncAbsurd) -> None:
    """The heart of durability: after a crash, replay serves the cached model response.

    Flow: spawn a task with `max_attempts=2`, enter the task context, make a request
    against the AbsurdModel (which stores a checkpoint), fail the run to force a retry,
    re-enter a fresh context against the same task, and verify the model is *not*
    called a second time but the response is identical.
    """
    counter = {'calls': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        counter['calls'] += 1
        return ModelResponse(parts=[TextPart(content=f'call {counter["calls"]}')])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:  # pragma: no cover
        counter['calls'] += 1
        yield f'call {counter["calls"]}'

    model = AbsurdModel(
        FunctionModel(fn, stream_function=stream_fn, model_name='fn'),
        step_name_prefix='crash',
    )

    async def noop(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:  # pragma: no cover
        return None

    absurd.register_task(name='crash')(noop)  # type: ignore[arg-type]

    # First attempt: record the checkpoint, then let the context manager exit (we
    # then explicitly fail the run so Absurd schedules a retry).
    spawned_id: str
    async with running_task_context(absurd, 'crash', max_attempts=2) as ctx:
        first_response = await model.request([], None, ModelRequestParameters())
        spawned_id = ctx.task_id

    # Second attempt: simulate a crash-and-retry, reclaim the task, and re-call the
    # model. The AbsurdModel must serve the cached response from the first attempt.
    async with reenter_running_task(absurd, spawned_id):
        replay_response = await model.request([], None, ModelRequestParameters())

    assert counter['calls'] == 1
    assert replay_response == first_response
    assert isinstance(replay_response.parts[0], TextPart)
    assert replay_response.parts[0].content == 'call 1'
