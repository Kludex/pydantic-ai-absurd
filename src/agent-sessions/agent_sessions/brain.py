from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar
from uuid import UUID

from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from pydantic_ai.agent import AgentRunResult

from ._pydantic_ai import messages_to_events
from .events import EventKind, Visibility
from .session import Session
from .wake import WakeHandle, wake as _wake_fn

AgentDepsT = TypeVar('AgentDepsT')


@dataclass(frozen=True)
class BrainInvocation:
    """Arguments an Absurd task passes to a brain handler at runtime."""

    session_id: UUID
    input: dict[str, JsonValue]
    causation_id: int | None = None


class _AbsurdAgentLike(Protocol):
    name: str | None

    async def run(
        self,
        user_prompt: str | None = None,
        *,
        message_history: list[Any] | None = None,
        deps: Any = None,
    ) -> AgentRunResult[Any]: ...


class BrainContext(Generic[AgentDepsT]):
    """The object passed to every `@brain`-decorated function.

    Encapsulates the live session, the Absurd task context, the raw wake() input,
    and convenience helpers for the common operations: run another agent against
    the session's history, append messages, and chain to other brains.
    """

    def __init__(
        self,
        *,
        name: str,
        absurd_ctx: AsyncTaskContext,
        absurd: AsyncAbsurd,
        session: Session,
        invocation: BrainInvocation,
    ) -> None:
        self._name = name
        self._absurd_ctx = absurd_ctx
        self._absurd = absurd
        self._session = session
        self._invocation = invocation
        self._actor = f'brain:{name}'
        # Every append from within a brain inherits the brain_started event as its
        # causation so downstream consumers can trace the chain.
        self._causation_id: int | None = invocation.causation_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def absurd_ctx(self) -> AsyncTaskContext:
        return self._absurd_ctx

    @property
    def session(self) -> Session:
        return self._session

    @property
    def input(self) -> dict[str, JsonValue]:
        return self._invocation.input

    async def agent_run(
        self,
        agent: _AbsurdAgentLike,
        user_prompt: str | None = None,
        *,
        deps: AgentDepsT | None = None,
    ) -> AgentRunResult[Any]:
        """Run a wrapped (AbsurdAgent) with this session's history, appending its new messages."""
        history = await self._session.messages()
        result = await agent.run(user_prompt, message_history=history, deps=deps)
        new_messages = list(result.new_messages())
        for kwargs in messages_to_events(new_messages, actor=self._actor, causation_id=self._causation_id):
            await self._session.append(**kwargs)
        return result

    async def post(self, content: str, *, visibility: Visibility = Visibility.public) -> None:
        """Append a plain assistant message - useful for progress text that isn't from an agent."""
        await self._session.append(
            kind=EventKind.assistant_message,
            actor=self._actor,
            payload={'content': content},
            visibility=visibility,
            causation_id=self._causation_id,
        )

    async def post_status(self, content: str) -> None:
        """Append a short status update. Distinguished from `post()` so UIs can render them differently."""
        await self._session.append(
            kind=EventKind.status_update,
            actor=self._actor,
            payload={'content': content},
            causation_id=self._causation_id,
        )

    async def sleep(self, seconds: float) -> None:
        """Durable sleep - survives worker restarts via Absurd's checkpoint."""
        await self._absurd_ctx.sleep_for(f'brain:{self._name}:sleep', seconds)

    async def wake(
        self,
        brain_name: str,
        *,
        input: dict[str, JsonValue] | None = None,
        dedup_key: str | None = None,
    ) -> WakeHandle:
        """Chain to another brain. Carries our own event sequence as the causation id."""
        return await _wake_fn(
            self._absurd,
            self._session,
            brain_name,
            input=input,
            dedup_key=dedup_key,
            causation_id=self._causation_id,
        )


@dataclass(frozen=True)
class BrainDefinition:
    """What the `@brain(name)` decorator captures - the name and the user handler."""

    name: str
    fn: Callable[[BrainContext[Any]], Awaitable[None]]


_REGISTRY: dict[str, BrainDefinition] = {}


def brain(name: str) -> Callable[[Callable[[BrainContext[Any]], Awaitable[None]]], BrainDefinition]:
    """Register a brain under `name`. The returned object captures the name and
    handler; actual Absurd task registration happens in `create_worker()`.
    """

    def decorator(fn: Callable[[BrainContext[Any]], Awaitable[None]]) -> BrainDefinition:
        if name in _REGISTRY:
            raise ValueError(f'brain {name!r} is already registered')
        definition = BrainDefinition(name=name, fn=fn)
        _REGISTRY[name] = definition
        return definition

    return decorator


def clear_registry() -> None:
    """Test-only helper to reset the module-level registry between tests."""
    _REGISTRY.clear()


def registered_brains() -> list[BrainDefinition]:
    return list(_REGISTRY.values())
