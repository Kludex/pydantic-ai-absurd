from __future__ import annotations

from absurd_sdk import AsyncTaskContext, get_current_context
from pydantic_ai.exceptions import UserError
from typing_extensions import TypedDict


class StepConfig(TypedDict, total=False):
    """Configuration applied to every Absurd step spawned by a wrapped agent."""

    max_attempts: int
    heartbeat_seconds: int


def current_async_context() -> AsyncTaskContext | None:
    """Return the current Absurd async task context, or None if not inside one."""
    ctx = get_current_context()
    if ctx is None:
        return None
    if isinstance(ctx, AsyncTaskContext):
        return ctx
    raise UserError('AbsurdAgent requires an async Absurd task context; got a sync one.')


def require_async_context() -> AsyncTaskContext:
    ctx = current_async_context()
    if ctx is None:
        raise UserError(
            'AbsurdAgent.run() must be called from inside an Absurd task handler '
            '(use `await absurd.spawn(...)` and have the task invoke the agent).'
        )
    return ctx
