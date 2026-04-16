from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar
from uuid import UUID

from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from psycopg import AsyncConnection
from psycopg.rows import TupleRow, dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic_ai.agent import AgentRunResult

from ._pydantic_ai import messages_to_events
from .events import EventKind, Visibility
from .session import Session

AsyncPool = AsyncConnectionPool[AsyncConnection[TupleRow]]
AgentDepsT = TypeVar('AgentDepsT')

Concurrency = Literal['queue', 'parallel', 'supersede']


@dataclass(frozen=True)
class WakeHandle:
    task_id: str
    dedup_key: str
    deduplicated: bool


@dataclass(frozen=True)
class BrainInvocation:
    session_id: UUID
    input: dict[str, JsonValue]
    causation_id: int | None = None


@dataclass(frozen=True)
class PoisonEvent:
    session_id: UUID
    brain_name: str
    task_id: str
    error: str


PoisonHandler = Callable[[PoisonEvent], Awaitable[None]]


class WakeDepthExceeded(RuntimeError):
    """Raised when a chain of `wake()` calls exceeds the configured max depth."""


class _AbsurdAgentLike(Protocol):
    name: str | None

    async def run(
        self,
        user_prompt: str | None = None,
        *,
        message_history: list[Any] | None = None,
        deps: Any = None,
    ) -> AgentRunResult[Any]: ...


BrainFn = Callable[['BrainContext[Any]'], Awaitable[None]]


@dataclass(frozen=True)
class BrainDefinition:
    name: str
    fn: BrainFn


class BrainContext(Generic[AgentDepsT]):
    """The object passed to every brain handler.

    Encapsulates the live session, the Absurd task context, the raw `wake()`
    input, and convenience helpers for the common operations: run another agent
    against the session's history, append messages, and chain to other brains
    on the same `Workflow`.
    """

    def __init__(
        self,
        *,
        workflow: Workflow,
        name: str,
        absurd_ctx: AsyncTaskContext,
        session: Session,
        invocation: BrainInvocation,
    ) -> None:
        self._workflow = workflow
        self._name = name
        self._absurd_ctx = absurd_ctx
        self._session = session
        self._invocation = invocation
        self._actor = f'brain:{name}'
        # Every append from within a brain inherits the brain_started event as
        # its causation so downstream consumers can trace the chain.
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
        """Append a plain assistant message."""
        await self._session.append(
            kind=EventKind.assistant_message,
            actor=self._actor,
            payload={'content': content},
            visibility=visibility,
            causation_id=self._causation_id,
        )

    async def post_status(self, content: str) -> None:
        """Append a short status update, distinguished from `post()` so UIs can render them differently."""
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
        """Chain to another brain on the same workflow. Carries our event as the causation id."""
        return await self._workflow.wake(
            self._session,
            brain_name,
            input=input,
            dedup_key=dedup_key,
            causation_id=self._causation_id,
        )


class Workflow:
    """Central object that holds brains, registers them with Absurd, and spawns wakes.

    Usage mirrors `Agent.tool` / `FastAPI.get`:

        workflow = Workflow(absurd=absurd, pool=pool)

        @workflow.brain('planner')
        async def planner(ctx):
            ...

        await workflow.wake(session, 'planner')
        await workflow.run()  # in a worker process

    Multiple workflows can coexist in the same process (e.g. different queues)
    - each holds its own brain registry. There is no module-level global state.
    """

    def __init__(
        self,
        *,
        absurd: AsyncAbsurd,
        pool: AsyncPool,
        max_wake_depth: int = 20,
        on_poison: PoisonHandler | None = None,
    ) -> None:
        self._absurd = absurd
        self._pool = pool
        self._max_wake_depth = max_wake_depth
        self._on_poison = on_poison
        self._brains: dict[str, BrainDefinition] = {}

    @property
    def absurd(self) -> AsyncAbsurd:
        return self._absurd

    @property
    def pool(self) -> AsyncPool:
        return self._pool

    @property
    def brains(self) -> list[BrainDefinition]:
        return list(self._brains.values())

    def brain(self, name: str) -> Callable[[BrainFn], BrainDefinition]:
        """Decorator: register a brain under `name` on this workflow and with Absurd."""

        def decorator(fn: BrainFn) -> BrainDefinition:
            if name in self._brains:
                raise ValueError(f'brain {name!r} is already registered on this workflow')
            definition = BrainDefinition(name=name, fn=fn)
            self._brains[name] = definition
            self._register_with_absurd(definition)
            return definition

        return decorator

    def task_name(self, brain_name: str) -> str:
        return f'brain.{brain_name}'

    async def run(self) -> None:  # pragma: no cover - exercised in examples
        """Start the Absurd polling loop until `stop()` is called."""
        await self._absurd.start_worker()

    def stop(self) -> None:  # pragma: no cover - counterpart to `run()`
        self._absurd.stop_worker()

    async def wake(
        self,
        session: Session | UUID,
        brain_name: str,
        *,
        input: dict[str, JsonValue] | None = None,
        dedup_key: str | None = None,
        concurrency: Concurrency = 'queue',
        causation_id: int | None = None,
        max_attempts: int | None = None,
    ) -> WakeHandle:
        """Idempotent brain trigger bound to this workflow's absurd client and pool.

        See module docstring for concurrency semantics.
        """
        session_id = session.id if isinstance(session, Session) else session
        input_payload: dict[str, JsonValue] = input or {}
        key = dedup_key or _deterministic_dedup_key(session_id, brain_name, input_payload)

        existing = await _lookup_dedup(self._pool, key)
        if existing is not None:
            return WakeHandle(task_id=existing, dedup_key=key, deduplicated=True)

        if concurrency == 'supersede':
            await _cancel_active_brains(self._absurd, self._pool, session_id, brain_name)

        params: dict[str, JsonValue] = {
            'session_id': str(session_id),
            'input': input_payload,
            'causation_id': causation_id,
            'concurrency': concurrency,
        }
        spawned = await self._absurd.spawn(
            self.task_name(brain_name),
            params,
            idempotency_key=key,
            max_attempts=max_attempts,
        )

        task_id = spawned['task_id']
        task_uuid = task_id if isinstance(task_id, UUID) else UUID(task_id)
        await _record_dedup(self._pool, key, session_id, brain_name, task_uuid)
        return WakeHandle(task_id=str(task_uuid), dedup_key=key, deduplicated=False)

    def _register_with_absurd(self, definition: BrainDefinition) -> None:
        workflow = self

        async def handler(params: Mapping[str, JsonValue] | None, ctx: AsyncTaskContext) -> JsonValue:
            return await _run_brain(workflow=workflow, definition=definition, params=params or {}, ctx=ctx)

        # Absurd's register_task decorator is typed for sync handlers; the
        # runtime dispatches to async handlers via `_execute_task`.
        self._absurd.register_task(name=self.task_name(definition.name))(handler)  # type: ignore[arg-type]


async def _run_brain(
    *,
    workflow: Workflow,
    definition: BrainDefinition,
    params: Mapping[str, JsonValue],
    ctx: AsyncTaskContext,
) -> JsonValue:
    session_id = _expect_uuid(params, 'session_id')
    raw_input = params.get('input') or {}
    if not isinstance(raw_input, dict):
        raise ValueError(f'brain input must be a JSON object, got {type(raw_input).__name__}')
    causation_id_raw = params.get('causation_id')
    causation_id: int | None = causation_id_raw if isinstance(causation_id_raw, int) else None
    concurrency_raw = params.get('concurrency')
    concurrency = concurrency_raw if isinstance(concurrency_raw, str) else 'queue'

    session = await Session.load(workflow.pool, session_id)
    await _check_wake_depth(workflow.pool, session_id, causation_id, workflow._max_wake_depth)

    # Single-active-brain enforcement when concurrency == 'queue': take a
    # session-level advisory lock for the duration of the brain body.
    if concurrency == 'queue':
        async with workflow.pool.connection() as conn:
            await conn.execute('SELECT pg_advisory_lock(hashtextextended(%s, 1))', (str(session_id),))
            try:
                await _execute_brain_body(
                    workflow=workflow,
                    definition=definition,
                    ctx=ctx,
                    session=session,
                    input_payload=raw_input,
                    causation_id=causation_id,
                )
            finally:
                await conn.execute('SELECT pg_advisory_unlock(hashtextextended(%s, 1))', (str(session_id),))
    else:
        await _execute_brain_body(
            workflow=workflow,
            definition=definition,
            ctx=ctx,
            session=session,
            input_payload=raw_input,
            causation_id=causation_id,
        )
    return None


async def _execute_brain_body(
    *,
    workflow: Workflow,
    definition: BrainDefinition,
    ctx: AsyncTaskContext,
    session: Session,
    input_payload: Mapping[str, JsonValue],
    causation_id: int | None,
) -> None:
    started = await session.append(
        kind=EventKind.brain_started,
        actor=f'brain:{definition.name}',
        payload={'brain': definition.name},
        visibility=Visibility.internal,
        causation_id=causation_id,
    )
    brain_context: BrainContext[Any] = BrainContext(
        workflow=workflow,
        name=definition.name,
        absurd_ctx=ctx,
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
        if workflow._on_poison is not None:
            await workflow._on_poison(
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
    """Follow `causation_id` back through the event chain, raising if too deep."""
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


def _expect_uuid(params: Mapping[str, JsonValue], key: str) -> UUID:
    value = params.get(key)
    if not isinstance(value, str):
        raise ValueError(f'brain params missing UUID field {key!r}')
    return UUID(value)


def _deterministic_dedup_key(session_id: UUID, brain_name: str, input_payload: dict[str, JsonValue]) -> str:
    digest = hashlib.sha256(json.dumps(input_payload, sort_keys=True).encode()).hexdigest()
    return f'{session_id}:{brain_name}:{digest}'


async def _lookup_dedup(pool: AsyncPool, key: str) -> str | None:
    async with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        await cur.execute('SELECT task_id FROM wake_dedup WHERE dedup_key = %s', (key,))
        row = await cur.fetchone()
    if row is None:
        return None
    return str(row['task_id'])


async def _record_dedup(pool: AsyncPool, key: str, session_id: UUID, brain_name: str, task_id: UUID) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO wake_dedup (dedup_key, session_id, brain_name, task_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (dedup_key) DO NOTHING
            """,
            (key, session_id, brain_name, task_id),
        )


async def _cancel_active_brains(
    absurd: AsyncAbsurd,
    pool: AsyncPool,
    session_id: UUID,
    brain_name: str,
) -> None:
    async with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        await cur.execute(
            'SELECT task_id FROM wake_dedup WHERE session_id = %s AND brain_name = %s',
            (session_id, brain_name),
        )
        rows = await cur.fetchall()
    for row in rows:
        await absurd.cancel_task(str(row['task_id']))
