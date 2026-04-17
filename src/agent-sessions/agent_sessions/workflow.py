from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar
from uuid import UUID, uuid4

import logfire
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue, SuspendTask
from psycopg import AsyncConnection
from psycopg.rows import TupleRow
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
    """Handle returned by `Workflow.wake()`.

    `task_id` is the Absurd task id - stable across duplicate wakes thanks to
    Absurd's native idempotency on `dedup_key`. Calling `wake()` twice with the
    same (implicit or explicit) `dedup_key` returns the same `task_id`; the
    caller can't tell "spawned new" from "deduplicated" without querying
    Absurd, which intentionally stays cheap.
    """

    task_id: str
    dedup_key: str


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
        session_lease_poll_seconds: float = 1.0,
    ) -> None:
        self._absurd = absurd
        self._pool = pool
        self._max_wake_depth = max_wake_depth
        self._on_poison = on_poison
        self._session_lease_poll_seconds = session_lease_poll_seconds
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

        Dedup is handled by Absurd's native `idempotency_key` - two wakes with the
        same `dedup_key` resolve to the same task_id without spawning twice.

        With `concurrency="supersede"`, any brain of the same name currently leased
        on this session is cancelled via `absurd.cancel_task` before we spawn. The
        cancelled task's cleanup releases the lease, so the new spawn can acquire
        it normally. Pending tasks that haven't yet taken the lease are not
        touched - the common use case ("user retried, replace what's running") only
        needs to cancel the live run.
        """
        session_id = session.id if isinstance(session, Session) else session
        input_payload: dict[str, JsonValue] = input or {}
        # Absurd's idempotency record outlives cancellation: if we reused the
        # same key after cancelling, spawn would return the cancelled task's id
        # instead of creating the replacement. Supersede explicitly wants a
        # *new* task every call, so we mint a fresh key and ignore any caller-
        # supplied dedup_key (dedup and supersede are mutually exclusive).
        if concurrency == 'supersede':
            key = f'{session_id}:{brain_name}:supersede:{uuid4()}'
        else:
            key = dedup_key or _deterministic_dedup_key(session_id, brain_name, input_payload)

        with logfire.span(
            'wake brain {brain_name}',
            brain_name=brain_name,
            session_id=str(session_id),
            concurrency=concurrency,
        ):
            if concurrency == 'supersede':
                await self._cancel_active_brain(session_id, brain_name)

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
            task_id_str = str(task_id) if isinstance(task_id, UUID) else task_id
            return WakeHandle(task_id=task_id_str, dedup_key=key)

    async def _cancel_active_brain(self, session_id: UUID, brain_name: str) -> None:
        """Cancel the Absurd task holding the lease for `(session_id, brain_name)`.

        We clear the lease row *before* firing `cancel_task`. A brain that's
        suspended inside `ctx.sleep_for` (or any other durable wait) won't run
        any Python cleanup when Absurd cancels it - there's no live stack to
        execute the `except BaseException: _release_session_lease(...)` branch
        in `_run_brain`. Clearing the lease here means a suspended-then-
        cancelled brain doesn't strand the session as "running" forever.

        We use a CTE so the clear and the read of the cancelled task_id
        happen in the same statement: no race where a just-released lease is
        re-taken between SELECT and UPDATE, and we always cancel at most the
        task we actually evicted. The CTE captures the pre-update row; the
        outer UPDATE wipes the lease.
        """
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                """
                WITH victim AS (
                    SELECT id, running_task_id
                    FROM agent_sessions.sessions
                    WHERE id = %s
                      AND running_brain_name = %s
                      AND running_task_id IS NOT NULL
                    FOR UPDATE
                )
                UPDATE agent_sessions.sessions AS s
                SET running_task_id = NULL, running_brain_name = NULL
                FROM victim
                WHERE s.id = victim.id
                RETURNING victim.running_task_id
                """,
                (session_id, brain_name),
            )
            row = await cur.fetchone()
        if row is None:
            return
        await self._absurd.cancel_task(str(row[0]))

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

    # Both 'queue' and 'supersede' enforce single-active-brain on the session via a
    # row-level lease on `agent_sessions.sessions.running_task_id`. 'supersede' also
    # takes the lease - otherwise a supersede'd task wouldn't write running_brain_name,
    # and a subsequent supersede call would find nothing to cancel. Postgres advisory
    # locks would be simpler but they're bound to the connection that acquired them,
    # so holding one for the brain's lifetime pins a pool connection - under concurrent
    # sessions that starves the pool and deadlocks against `session.append()` on the
    # conn the brain is itself trying to use. The lease pattern only touches the pool
    # for the brief CAS queries and never holds a conn between them.
    if concurrency in ('queue', 'supersede'):
        await _acquire_session_lease(
            workflow.pool, session_id, definition.name, ctx, workflow._session_lease_poll_seconds
        )
        try:
            await _execute_brain_body(
                workflow=workflow,
                definition=definition,
                ctx=ctx,
                session=session,
                input_payload=raw_input,
                causation_id=causation_id,
            )
        except SuspendTask:
            # A durable sleep (ctx.sleep_for / await_event) isn't a brain finishing -
            # the task will resume later and re-enter _run_brain. Keep the lease held
            # so nothing barges in during the wait (and so `supersede` can still find
            # a live lease to cancel).
            raise
        except BaseException:
            await _release_session_lease(workflow.pool, session_id, ctx.task_id)
            raise
        await _release_session_lease(workflow.pool, session_id, ctx.task_id)
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


_LEASE_POLL_STEP = 'agent_sessions:session_lease_poll'


async def _acquire_session_lease(
    pool: AsyncPool,
    session_id: UUID,
    brain_name: str,
    ctx: AsyncTaskContext,
    poll_seconds: float,
) -> None:
    """Acquire the per-session lease, suspending via Absurd when another brain holds it.

    Each poll is a single, briefly-held pool connection. Between polls the Absurd task
    sleeps durably via `ctx.sleep_for(...)`, so a contended session doesn't keep a pool
    connection pinned and doesn't count against the worker's concurrency budget.

    The lease records both the task_id (lease owner) and the brain_name (so `supersede`
    can target the right brain). If this is a retry of a previous attempt that crashed
    without releasing, the stale lease still points at our own task_id - we clear it
    ourselves before the CAS, which is safe because only this task is running right now
    (Absurd only redelivers a run once the previous one is marked failed).
    """
    # ctx.task_id is typed `str` in absurd-sdk but arrives as a `uuid.UUID` at
    # runtime; stringify so the TEXT-column comparison doesn't hit
    # `operator does not exist: text = uuid`.
    lease_owner = str(ctx.task_id)
    async with pool.connection() as conn:
        await conn.execute(
            'UPDATE agent_sessions.sessions '
            'SET running_task_id = NULL, running_brain_name = NULL '
            'WHERE id = %s AND running_task_id = %s',
            (session_id, lease_owner),
        )
    while True:
        async with pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE agent_sessions.sessions
                SET running_task_id = %s, running_brain_name = %s
                WHERE id = %s AND running_task_id IS NULL
                RETURNING 1
                """,
                (lease_owner, brain_name, session_id),
            )
            acquired = await cur.fetchone() is not None
        if acquired:
            return
        await ctx.sleep_for(_LEASE_POLL_STEP, poll_seconds)


async def _release_session_lease(pool: AsyncPool, session_id: UUID, task_id: str) -> None:
    """Release the lease. The `running_task_id = ?` guard prevents us from clearing a
    lease that's been force-taken by another task after a poison-cleanup."""
    lease_owner = str(task_id)
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE agent_sessions.sessions
            SET running_task_id = NULL, running_brain_name = NULL
            WHERE id = %s AND running_task_id = %s
            """,
            (session_id, lease_owner),
        )


async def _execute_brain_body(
    *,
    workflow: Workflow,
    definition: BrainDefinition,
    ctx: AsyncTaskContext,
    session: Session,
    input_payload: Mapping[str, JsonValue],
    causation_id: int | None,
) -> None:
    with logfire.span(
        'run brain {brain_name}',
        brain_name=definition.name,
        session_id=str(session.id),
        task_id=str(ctx.task_id),
    ) as span:
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
        except SuspendTask:
            # Durable sleep / await_event: the task will resume and re-enter the
            # brain, don't treat this as a failure.
            span.set_attribute('outcome', 'suspended')
            raise
        except BaseException as exc:
            span.record_exception(exc)
            span.set_attribute('outcome', 'failed')
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

        span.set_attribute('outcome', 'finished')
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
                'SELECT causation_id FROM agent_sessions.session_events WHERE session_id = %s AND sequence = %s',
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
