from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from .brain import (
    BrainContext,
    BrainDefinition,
    BrainInvocation,
    registered_brains,
)
from .events import EventKind, Visibility
from .session import Session
from .wake import brain_task_name

AsyncPool = AsyncConnectionPool[AsyncConnection[TupleRow]]


@dataclass(frozen=True)
class PoisonEvent:
    session_id: UUID
    brain_name: str
    task_id: str
    error: str


PoisonHandler = Callable[[PoisonEvent], Awaitable[None]]


class Worker:
    """Thin wrapper around `AsyncAbsurd` that registers every known brain and runs the loop."""

    def __init__(
        self,
        *,
        absurd: AsyncAbsurd,
        pool: AsyncPool,
        brains: list[BrainDefinition],
        max_wake_depth: int = 20,
        on_poison: PoisonHandler | None = None,
    ) -> None:
        self._absurd = absurd
        self._pool = pool
        self._brains = list(brains)
        self._max_wake_depth = max_wake_depth
        self._on_poison = on_poison
        self._registered = False

    def register(self) -> None:
        """Register every brain as an Absurd task. Idempotent."""
        if self._registered:
            return
        self._registered = True
        for definition in self._brains:
            self._register_one(definition)

    def _register_one(self, definition: BrainDefinition) -> None:
        absurd = self._absurd
        pool = self._pool
        max_depth = self._max_wake_depth
        on_poison = self._on_poison

        async def handler(params: Mapping[str, JsonValue] | None, ctx: AsyncTaskContext) -> JsonValue:
            return await _run_brain(
                definition=definition,
                params=params or {},
                ctx=ctx,
                absurd=absurd,
                pool=pool,
                max_depth=max_depth,
                on_poison=on_poison,
            )

        # Absurd's register_task decorator is typed for sync handlers; the runtime
        # dispatches to async handlers via `_execute_task` (see pydantic-ai-absurd).
        absurd.register_task(name=brain_task_name(definition.name))(handler)  # type: ignore[arg-type]

    async def run(self) -> None:  # pragma: no cover - exercised in examples, not unit tests
        """Start the Absurd polling loop until `stop()` is called."""
        self.register()
        await self._absurd.start_worker()

    def stop(self) -> None:  # pragma: no cover - counterpart to `run()`
        self._absurd.stop_worker()


async def create_worker(
    *,
    absurd: AsyncAbsurd,
    pool: AsyncPool,
    brains: list[BrainDefinition] | None = None,
    max_wake_depth: int = 20,
    on_poison: PoisonHandler | None = None,
) -> Worker:
    """Register every brain provided (or every brain in the module-level registry) and return a Worker.

    The returned Worker has not started polling yet - call `await worker.run()` to
    start the Absurd loop, or `worker.register()` if you want the tasks known to
    Absurd but plan to drive the queue yourself (useful in tests).
    """
    worker = Worker(
        absurd=absurd,
        pool=pool,
        brains=brains if brains is not None else registered_brains(),
        max_wake_depth=max_wake_depth,
        on_poison=on_poison,
    )
    worker.register()
    return worker


async def _run_brain(
    *,
    definition: BrainDefinition,
    params: Mapping[str, JsonValue],
    ctx: AsyncTaskContext,
    absurd: AsyncAbsurd,
    pool: AsyncPool,
    max_depth: int,
    on_poison: PoisonHandler | None,
) -> JsonValue:
    session_id = _expect_uuid(params, 'session_id')
    raw_input = params.get('input') or {}
    if not isinstance(raw_input, dict):
        raise ValueError(f'brain input must be a JSON object, got {type(raw_input).__name__}')
    causation_id_raw = params.get('causation_id')
    causation_id: int | None = causation_id_raw if isinstance(causation_id_raw, int) else None
    concurrency_raw = params.get('concurrency')
    concurrency = concurrency_raw if isinstance(concurrency_raw, str) else 'queue'

    session = await Session.load(pool, session_id)

    await _check_wake_depth(pool, session_id, causation_id, max_depth)

    # Single-active-brain enforcement when concurrency == 'queue': take a
    # transaction-scoped advisory lock keyed by the session id. The lock is held
    # for the lifetime of the brain's database operations - which in practice
    # means the lifetime of the brain, because the brain does all its state
    # mutation via the session.
    if concurrency == 'queue':
        async with pool.connection() as conn:
            await conn.execute(
                'SELECT pg_advisory_lock(hashtextextended(%s, 1))',
                (str(session_id),),
            )
            try:
                await _execute_brain_body(
                    definition=definition,
                    ctx=ctx,
                    absurd=absurd,
                    session=session,
                    input_payload=raw_input,
                    causation_id=causation_id,
                    on_poison=on_poison,
                )
            finally:
                await conn.execute(
                    'SELECT pg_advisory_unlock(hashtextextended(%s, 1))',
                    (str(session_id),),
                )
    else:
        await _execute_brain_body(
            definition=definition,
            ctx=ctx,
            absurd=absurd,
            session=session,
            input_payload=raw_input,
            causation_id=causation_id,
            on_poison=on_poison,
        )
    return None


async def _execute_brain_body(
    *,
    definition: BrainDefinition,
    ctx: AsyncTaskContext,
    absurd: AsyncAbsurd,
    session: Session,
    input_payload: Mapping[str, JsonValue],
    causation_id: int | None,
    on_poison: PoisonHandler | None,
) -> None:
    started = await session.append(
        kind=EventKind.brain_started,
        actor=f'brain:{definition.name}',
        payload={'brain': definition.name},
        visibility=Visibility.internal,
        causation_id=causation_id,
    )
    brain_context: BrainContext[Any] = BrainContext(
        name=definition.name,
        absurd_ctx=ctx,
        absurd=absurd,
        session=session,
        invocation=BrainInvocation(
            session_id=session.id,
            input=dict(input_payload),
            causation_id=started.sequence,
        ),
    )

    try:
        await definition.fn(brain_context)
    except BaseException as exc:
        await session.append(
            kind=EventKind.brain_failed,
            actor=f'brain:{definition.name}',
            payload={'error': repr(exc)},
            visibility=Visibility.internal,
            causation_id=started.sequence,
        )
        if on_poison is not None:
            await on_poison(
                PoisonEvent(
                    session_id=session.id,
                    brain_name=definition.name,
                    task_id=ctx.task_id,
                    error=repr(exc),
                )
            )
        raise

    await session.append(
        kind=EventKind.brain_finished,
        actor=f'brain:{definition.name}',
        payload={'brain': definition.name},
        visibility=Visibility.internal,
        causation_id=started.sequence,
    )


async def _check_wake_depth(
    pool: AsyncPool,
    session_id: UUID,
    causation_id: int | None,
    max_depth: int,
) -> None:
    """Follow `causation_id` back through the event chain, raising if too deep.

    A chain is a sequence of brain_started events each caused by the previous one.
    We bail the moment we either exceed `max_depth` or run off the start.
    """
    if causation_id is None:
        return
    depth = 0
    current: int | None = causation_id
    async with pool.connection() as conn:
        while current is not None:
            cur = conn.cursor()
            await cur.execute(
                'SELECT causation_id FROM session_events WHERE session_id = %s AND sequence = %s',
                (session_id, current),
            )
            row = await cur.fetchone()
            if row is None:
                return
            depth += 1
            if depth > max_depth:
                raise WakeDepthExceeded(f'wake chain exceeded max depth {max_depth} for session {session_id}')
            current = row[0]


class WakeDepthExceeded(RuntimeError):
    """Raised when a chain of `wake()` calls exceeds the configured max depth."""


def _expect_uuid(params: Mapping[str, JsonValue], key: str) -> UUID:
    value = params.get(key)
    if not isinstance(value, str):
        raise ValueError(f'brain params missing UUID field {key!r}')
    return UUID(value)
