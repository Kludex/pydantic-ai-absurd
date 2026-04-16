from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter
from pydantic_ai import ModelMessage
from pydantic_ai.agent import AgentRunResult

from .events import EventKind, SessionEvent, Visibility

_message_adapter: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)
_messages_adapter: TypeAdapter[list[ModelMessage]] = TypeAdapter(list[ModelMessage])

USER_KINDS: frozenset[str] = frozenset({EventKind.user_message, EventKind.assistant_message})


def serialize_message(message: ModelMessage) -> dict[str, Any]:
    dumped = _message_adapter.dump_python(message, mode='json')
    assert isinstance(dumped, dict)
    return dumped


def deserialize_message(payload: dict[str, Any]) -> ModelMessage:
    return _message_adapter.validate_python(payload)


def messages_to_events(
    messages: list[ModelMessage],
    *,
    actor: str,
    causation_id: int | None = None,
) -> list[dict[str, Any]]:
    """Project a list of ModelMessage into the append kwargs for `Session.append()`.

    Each message becomes one event. `ModelRequest` messages carrying a user prompt are
    stored as `user_message` (public); `ModelResponse` messages become
    `assistant_message` (public); anything else (tool returns inside requests,
    tool-call-only responses) becomes internal-visibility so the UI stream stays
    clean while agents still see the full history via `message_history=`.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        kind, visibility = _classify(msg)
        out.append(
            {
                'kind': kind,
                'actor': actor,
                'payload': serialize_message(msg),
                'visibility': visibility,
                'causation_id': causation_id,
            }
        )
    return out


def _classify(msg: ModelMessage) -> tuple[str, Visibility]:
    if msg.kind == 'request':
        if any(part.part_kind == 'user-prompt' for part in msg.parts):
            return EventKind.user_message, Visibility.public
        return EventKind.tool_result, Visibility.internal
    if any(part.part_kind == 'text' for part in msg.parts):
        return EventKind.assistant_message, Visibility.public
    return EventKind.tool_call, Visibility.internal


def events_to_messages(events: list[SessionEvent]) -> list[ModelMessage]:
    """Rebuild a list of `ModelMessage` from session events.

    Events whose payload isn't a serialized `ModelMessage` (e.g. `status_update`,
    lifecycle events) are skipped - they exist for the UI, not the agent.
    """
    messages: list[ModelMessage] = []
    for event in events:
        if event.kind in (
            EventKind.user_message,
            EventKind.assistant_message,
            EventKind.tool_call,
            EventKind.tool_result,
        ):
            try:
                messages.append(deserialize_message(event.payload))
            except (ValueError, TypeError):  # pragma: no cover - malformed payload is a developer error
                continue
    return messages


def agent_run_result_to_events(
    result: AgentRunResult[Any],
    *,
    actor: str,
    causation_id: int | None = None,
) -> list[dict[str, Any]]:
    """Shortcut: project the new messages from an agent run into append kwargs."""
    return messages_to_events(list(result.new_messages()), actor=actor, causation_id=causation_id)
