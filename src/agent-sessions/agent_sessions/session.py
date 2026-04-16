from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import anyio
from psycopg import AsyncConnection
from psycopg.rows import TupleRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic_ai import ModelMessage

from ._pydantic_ai import events_to_messages
from .events import SessionEvent, Snapshot, Visibility

AsyncPool = AsyncConnectionPool[AsyncConnection[TupleRow]]


@dataclass(frozen=True)
class SessionHandle:
    """Lightweight identity for a session - an `id` plus the pool used to talk to it."""

    id: UUID
    pool: AsyncPool


class Session:
    """Durable conversation log stored in Postgres.

    Each session has an append-only event stream in `session_events`. Appends are
    serialized per-session via a transaction-scoped advisory lock (`session_id`
    hashed into a bigint), so concurrent appenders never race on the sequence
    number even though the table has no natural unique key on `sequence` alone.
    """

    def __init__(self, pool: AsyncPool, session_id: UUID) -> None:
        self._pool = pool
        self._id = session_id

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def pool(self) -> AsyncPool:
        return self._pool

    @classmethod
    async def create(
        cls,
        pool: AsyncPool,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        session_id = uuid4()
        async with pool.connection() as conn:
            await conn.execute(
                'INSERT INTO sessions (id, metadata) VALUES (%s, %s)',
                (session_id, Jsonb(metadata or {})),
            )
        return cls(pool, session_id)

    @classmethod
    async def load(cls, pool: AsyncPool, session_id: UUID) -> Session:
        async with pool.connection() as conn:
            cur = await conn.execute('SELECT 1 FROM sessions WHERE id = %s', (session_id,))
            row = await cur.fetchone()
            if row is None:
                raise LookupError(f'session {session_id} not found')
        return cls(pool, session_id)

    async def append(
        self,
        *,
        kind: str,
        actor: str,
        payload: dict[str, Any],
        visibility: Visibility = Visibility.public,
        payload_version: int = 1,
        causation_id: int | None = None,
        supersedes: int | None = None,
    ) -> SessionEvent:
        """Append an event. Serialized per-session via an advisory lock."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                # Advisory lock keyed by session UUID ensures two appenders on the same
                # session serialize; different sessions remain fully parallel.
                await conn.execute(
                    'SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))',
                    (str(self._id),),
                )
                cur = conn.cursor(row_factory=dict_row)
                await cur.execute(
                    'SELECT COALESCE(MAX(sequence), 0) + 1 AS next_seq FROM session_events WHERE session_id = %s',
                    (self._id,),
                )
                seq_row = await cur.fetchone()
                assert seq_row is not None
                sequence = int(seq_row['next_seq'])

                await cur.execute(
                    """
                    INSERT INTO session_events
                        (session_id, sequence, kind, actor, visibility, payload_version,
                         payload, causation_id, supersedes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING created_at
                    """,
                    (
                        self._id,
                        sequence,
                        kind,
                        actor,
                        visibility.value,
                        payload_version,
                        Jsonb(payload),
                        causation_id,
                        supersedes,
                    ),
                )
                created = await cur.fetchone()
                assert created is not None
                created_at = created['created_at']

                await conn.execute(
                    'UPDATE sessions SET updated_at = now() WHERE id = %s',
                    (self._id,),
                )
                # Fire pg_notify for anyone in listen(); fired inside the same transaction
                # so listeners don't see the notify before the row is visible.
                await conn.execute(
                    'SELECT pg_notify(%s, %s)',
                    (f'session_{self._id.hex}', str(sequence)),
                )

        return SessionEvent(
            session_id=self._id,
            sequence=sequence,
            kind=kind,
            actor=actor,
            visibility=visibility,
            payload_version=payload_version,
            payload=payload,
            causation_id=causation_id,
            supersedes=supersedes,
            created_at=created_at,
        )

    async def events(
        self,
        *,
        after: int = 0,
        limit: int | None = None,
        visibility: Visibility | None = None,
    ) -> list[SessionEvent]:
        query = [
            'SELECT session_id, sequence, kind, actor, visibility, payload_version,',
            '       payload, causation_id, supersedes, created_at',
            'FROM session_events',
            'WHERE session_id = %s AND sequence > %s',
        ]
        params: list[Any] = [self._id, after]
        if visibility is not None:
            query.append('AND visibility = %s')
            params.append(visibility.value)
        query.append('ORDER BY sequence ASC')
        if limit is not None:
            query.append('LIMIT %s')
            params.append(limit)

        async with self._pool.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(' '.join(query), tuple(params))
            rows = await cur.fetchall()
        return [SessionEvent.model_validate(row) for row in rows]

    async def latest_snapshot(self) -> Snapshot | None:
        async with self._pool.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                """
                SELECT session_id, up_to_sequence, summary_payload, created_at
                FROM session_snapshots
                WHERE session_id = %s
                ORDER BY up_to_sequence DESC
                LIMIT 1
                """,
                (self._id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return Snapshot.model_validate(row)

    async def create_snapshot(self, *, up_to_sequence: int, summary_payload: dict[str, Any]) -> Snapshot:
        async with self._pool.connection() as conn:
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                """
                INSERT INTO session_snapshots (session_id, up_to_sequence, summary_payload)
                VALUES (%s, %s, %s)
                RETURNING created_at
                """,
                (self._id, up_to_sequence, Jsonb(summary_payload)),
            )
            row = await cur.fetchone()
            assert row is not None
        return Snapshot(
            session_id=self._id,
            up_to_sequence=up_to_sequence,
            summary_payload=summary_payload,
            created_at=row['created_at'],
        )

    async def messages(self) -> list[ModelMessage]:
        """User-facing view ready to pass as `message_history=` to a Pydantic AI agent.

        Starts from the latest snapshot (if any) and applies events after it.
        Events that don't correspond to a `ModelMessage` (e.g. `status_update` or
        lifecycle events) are filtered out here - agents only see what they can
        consume.
        """
        snapshot = await self.latest_snapshot()
        starting_from = snapshot.up_to_sequence if snapshot is not None else 0
        events = await self.events(after=starting_from)
        return events_to_messages(events)

    def listen(
        self,
        *,
        conninfo: str,
        timeout: float = 30.0,
        stop_after: int | None = None,
        ready: anyio.Event | None = None,
    ) -> AsyncIterator[SessionEvent]:  # pragma: no cover - experimental, covered by an integration example
        """Async iterator over new events via `LISTEN/NOTIFY`.

        Takes a `conninfo` (DSN) rather than the pool because LISTEN requires a
        dedicated autocommit connection held for the iterator's lifetime - it
        can't share a pooled connection.

        Pass an `anyio.Event` as `ready` and it will be `set()` immediately after
        the `LISTEN` statement has been issued, so writers can reliably wait
        without racy sleeps.

        pg_notify delivery is lossy - consumers should reconcile gaps with
        incremental `events(after=...)` reads on reconnect. `stop_after` is a
        test-only knob to bound the iterator at N events; in production, callers
        should manage lifetime via `anyio.CancelScope` or similar.
        """
        return _listen_iterator(conninfo, self._id, timeout, stop_after, ready)


async def _listen_iterator(  # pragma: no cover - experimental; covered by an integration example
    conninfo: str,
    session_id: UUID,
    timeout: float,
    stop_after: int | None,
    ready: anyio.Event | None,
) -> AsyncIterator[SessionEvent]:
    """LISTEN on a session channel and yield SessionEvent rows looked up by notified sequence."""
    channel = f'session_{session_id.hex}'
    async with await AsyncConnection.connect(conninfo, autocommit=True) as conn:
        await conn.execute(f'LISTEN {channel}')
        if ready is not None:
            ready.set()
        yielded = 0
        gen = conn.notifies(timeout=timeout, stop_after=stop_after)
        async for notify in gen:
            seq = int(notify.payload)
            cur = conn.cursor(row_factory=dict_row)
            await cur.execute(
                """
                SELECT session_id, sequence, kind, actor, visibility, payload_version,
                       payload, causation_id, supersedes, created_at
                FROM session_events
                WHERE session_id = %s AND sequence = %s
                """,
                (session_id, seq),
            )
            row = await cur.fetchone()
            if row is not None:
                yield SessionEvent.model_validate(row)
                yielded += 1
                if stop_after is not None and yielded >= stop_after:
                    return
