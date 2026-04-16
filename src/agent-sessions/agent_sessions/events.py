from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EventKind(StrEnum):
    """Open enum of event kinds. Libraries embedding this can add their own via plain strings."""

    user_message = 'user_message'
    assistant_message = 'assistant_message'
    tool_call = 'tool_call'
    tool_result = 'tool_result'
    status_update = 'status_update'
    brain_started = 'brain_started'
    brain_finished = 'brain_finished'
    brain_failed = 'brain_failed'
    snapshot_created = 'snapshot_created'


class Visibility(StrEnum):
    public = 'public'
    internal = 'internal'


class SessionEvent(BaseModel):
    """An immutable session event. Serialized as the row in `session_events`.

    `kind` is typed as a plain `str` so library users can introduce their own event
    kinds without forking the schema; canonical kinds live in `EventKind` and
    consumers should treat unknown kinds as opaque.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    session_id: UUID
    sequence: int
    kind: str
    actor: str
    visibility: Visibility = Visibility.public
    payload_version: int = 1
    payload: dict[str, Any] = Field(default_factory=dict)
    causation_id: int | None = None
    supersedes: int | None = None
    created_at: datetime


class Snapshot(BaseModel):
    """A compaction snapshot - collapses events up to `up_to_sequence` into `summary_payload`."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    session_id: UUID
    up_to_sequence: int
    summary_payload: dict[str, Any]
    created_at: datetime
