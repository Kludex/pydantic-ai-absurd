"""End-to-end proof that a brain's LLM calls are replayed from the Absurd
checkpoint on retry - the whole point of the package.

Flow:
  1. Spawn a brain that (a) calls an AbsurdAgent, (b) raises on the first
     attempt only, (c) posts 'done' on the second attempt.
  2. Drive Absurd twice (work_batch x2).
  3. Assert the inner model's FunctionModel body ran once, not twice.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic_ai import Agent, ModelMessage, ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_absurd import AbsurdAgent

from agent_sessions import BrainContext, EventKind, Session, Workflow

pytestmark = pytest.mark.anyio


async def test_brain_replay_does_not_recall_the_model(workflow: Workflow) -> None:
    model_calls = {'count': 0}
    attempts = {'count': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_calls['count'] += 1
        return ModelResponse(parts=[TextPart(content='pong')])

    async def stream_fn(  # pragma: no cover - not driven here
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        model_calls['count'] += 1
        yield 'pong'

    inner = Agent(
        FunctionModel(fn, stream_function=stream_fn, model_name='m'),
        name='inner',
    )
    absurd_agent = AbsurdAgent(inner, workflow.absurd)

    @workflow.brain('flaky')
    async def flaky(ctx: BrainContext[None]) -> None:
        attempts['count'] += 1
        # First attempt: call the model (which commits a checkpoint), then raise.
        # Second attempt: the model call hits the cached checkpoint and returns
        # the same response without re-running `fn`.
        result = await ctx.agent_run(absurd_agent, 'what?')
        if attempts['count'] == 1:
            raise RuntimeError('simulated crash after first model call')
        await ctx.post(f'replayed: {result.output}')

    session = await Session.create(workflow.pool)
    await workflow.wake(session, 'flaky', max_attempts=2)

    # Drain twice: first attempt fails, Absurd schedules a retry, second
    # attempt reads the checkpoint and completes.
    for _ in range(4):
        await workflow.absurd.work_batch(batch_size=1)

    # The LLM (FunctionModel body) must have been called exactly once.
    assert model_calls['count'] == 1, f'model was called {model_calls["count"]} times, expected 1'
    assert attempts['count'] == 2

    events = await session.events()
    kinds = [e.kind for e in events]
    # Attempt 1 records brain_started + its messages + brain_failed. Absurd's
    # retry fires the brain again, which calls the model (cached), then posts
    # 'replayed: pong' and emits brain_finished.
    assert EventKind.brain_failed in kinds
    assert EventKind.brain_finished in kinds
    final_text = next(
        e.payload['content']
        for e in reversed(events)
        if e.kind == EventKind.assistant_message and 'replayed' in e.payload.get('content', '')
    )
    assert final_text == 'replayed: pong'
