from __future__ import annotations

import pytest
from pydantic_ai import Agent, ModelMessage
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_sessions import EventKind, Session, Visibility
from agent_sessions._pydantic_ai import (
    agent_run_result_to_events,
    deserialize_message,
    messages_to_events,
    serialize_message,
)

from .conftest import AsyncPool

pytestmark = pytest.mark.anyio


def test_user_message_round_trip() -> None:
    msg = ModelRequest(parts=[UserPromptPart(content='hi')])
    assert deserialize_message(serialize_message(msg)) == msg


def test_messages_to_events_classifies_user_message() -> None:
    msg = ModelRequest(parts=[UserPromptPart(content='hi')])
    events = messages_to_events([msg], actor='user')
    assert len(events) == 1
    assert events[0]['kind'] == EventKind.user_message
    assert events[0]['visibility'] == Visibility.public


def test_messages_to_events_classifies_tool_return_as_internal() -> None:
    msg = ModelRequest(parts=[ToolReturnPart(tool_name='x', content='result', tool_call_id='c1')])
    events = messages_to_events([msg], actor='brain:x')
    assert events[0]['kind'] == EventKind.tool_result
    assert events[0]['visibility'] == Visibility.internal


def test_messages_to_events_classifies_assistant_message() -> None:
    msg = ModelResponse(parts=[TextPart(content='hello')])
    events = messages_to_events([msg], actor='brain:x')
    assert events[0]['kind'] == EventKind.assistant_message
    assert events[0]['visibility'] == Visibility.public


def test_messages_to_events_classifies_tool_call_only_as_internal() -> None:
    msg = ModelResponse(parts=[ToolCallPart(tool_name='x', args={}, tool_call_id='c1')])
    events = messages_to_events([msg], actor='brain:x')
    assert events[0]['kind'] == EventKind.tool_call
    assert events[0]['visibility'] == Visibility.internal


async def test_agent_run_result_to_events(pool: AsyncPool) -> None:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='ok')])

    agent = Agent(FunctionModel(fn, model_name='m'), name='a')
    result = await agent.run('hello')
    projected = agent_run_result_to_events(result, actor='brain:a', causation_id=7)
    assert any(ev['causation_id'] == 7 for ev in projected)
    assert any(ev['actor'] == 'brain:a' for ev in projected)


async def test_events_to_messages_round_trip(pool: AsyncPool) -> None:
    session = await Session.create(pool)
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hi')]),
        ModelResponse(parts=[TextPart(content='hello')]),
    ]
    for kwargs in messages_to_events(original, actor='user'):
        await session.append(**kwargs)
    restored = await session.messages()
    assert restored == original


async def test_events_to_messages_skips_non_message_events(pool: AsyncPool) -> None:
    session = await Session.create(pool)
    hi = serialize_message(ModelRequest(parts=[UserPromptPart(content='hi')]))
    await session.append(kind='user_message', actor='user', payload=hi)
    await session.append(kind='status_update', actor='brain:x', payload={'text': 'thinking...'})
    messages = await session.messages()
    assert len(messages) == 1


async def test_session_messages_uses_snapshot_as_floor(pool: AsyncPool) -> None:
    session = await Session.create(pool)
    old = serialize_message(ModelRequest(parts=[UserPromptPart(content='old')]))
    older_reply = serialize_message(ModelResponse(parts=[TextPart(content='older reply')]))
    new = serialize_message(ModelRequest(parts=[UserPromptPart(content='new')]))
    await session.append(kind='user_message', actor='user', payload=old)
    await session.append(kind='assistant_message', actor='brain:x', payload=older_reply)
    await session.create_snapshot(up_to_sequence=2, summary_payload={'summary': 'two'})
    await session.append(kind='user_message', actor='user', payload=new)
    messages = await session.messages()
    # Snapshot masks the first two; only the post-snapshot message is visible.
    assert len(messages) == 1
    assert isinstance(messages[0], ModelRequest)
    prompt = messages[0].parts[0]
    assert isinstance(prompt, UserPromptPart)
    assert prompt.content == 'new'
