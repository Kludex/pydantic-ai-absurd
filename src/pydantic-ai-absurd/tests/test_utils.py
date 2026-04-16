from __future__ import annotations

from typing import Any

import pytest
from absurd_sdk import TaskContext, _current_task_context
from pydantic_ai.exceptions import UserError

from pydantic_ai_absurd._utils import current_async_context, require_async_context

pytestmark = pytest.mark.anyio


def test_no_context_returns_none() -> None:
    assert current_async_context() is None


def test_require_async_context_raises_when_missing() -> None:
    with pytest.raises(UserError, match='must be called from inside an Absurd task handler'):
        require_async_context()


def test_sync_context_raises() -> None:
    # Construct a TaskContext (sync) bypassing its __init__ which normally forbids this.
    dummy: Any = object.__new__(TaskContext)
    token = _current_task_context.set(dummy)
    try:
        with pytest.raises(UserError, match='async Absurd task context'):
            current_async_context()
    finally:
        _current_task_context.reset(token)
