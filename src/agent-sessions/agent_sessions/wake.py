from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from absurd_sdk import AsyncAbsurd, JsonValue
from psycopg import AsyncConnection
from psycopg.rows import TupleRow, dict_row
from psycopg_pool import AsyncConnectionPool

from .session import Session


def brain_task_name(name: str) -> str:
    return f'brain.{name}'


AsyncPool = AsyncConnectionPool[AsyncConnection[TupleRow]]

Concurrency = Literal['queue', 'parallel', 'supersede']


@dataclass(frozen=True)
class WakeHandle:
    task_id: str
    dedup_key: str
    deduplicated: bool


async def wake(
    absurd: AsyncAbsurd,
    session: Session | UUID,
    brain_name: str,
    *,
    input: dict[str, JsonValue] | None = None,
    dedup_key: str | None = None,
    concurrency: Concurrency = 'queue',
    causation_id: int | None = None,
    pool: AsyncPool | None = None,
    max_attempts: int | None = None,
) -> WakeHandle:
    """Idempotent brain trigger.

    If `dedup_key` is given and a wake with the same key already exists for this
    session+brain, the existing task handle is returned without spawning a new
    task. If `dedup_key` is omitted, a deterministic key is computed from
    `(session_id, brain_name, sha256(json(input)))`.

    Concurrency policies:
      - `queue`     (default) - if the session already has an active brain, run
        this one after it finishes. Enforced by the brain handler taking a
        session-level advisory lock before doing any work.
      - `parallel`  - no locking; the new brain runs immediately alongside.
      - `supersede` - cancel the active brain, then spawn this one.
    """
    session_id = session.id if isinstance(session, Session) else session
    effective_pool = pool or (session.pool if isinstance(session, Session) else None)
    if effective_pool is None:
        raise ValueError('wake() requires either a Session or an explicit pool= argument')

    input_payload: dict[str, JsonValue] = input or {}
    key = dedup_key or _deterministic_dedup_key(session_id, brain_name, input_payload)

    existing = await _lookup_dedup(effective_pool, key)
    if existing is not None:
        return WakeHandle(task_id=existing, dedup_key=key, deduplicated=True)

    if concurrency == 'supersede':
        await _cancel_active_brains(absurd, effective_pool, session_id, brain_name)

    task_name = brain_task_name(brain_name)
    params: dict[str, JsonValue] = {
        'session_id': str(session_id),
        'input': input_payload,
        'causation_id': causation_id,
        'concurrency': concurrency,
    }
    spawned = await absurd.spawn(task_name, params, idempotency_key=key, max_attempts=max_attempts)

    task_id = spawned['task_id']
    task_uuid = task_id if isinstance(task_id, UUID) else UUID(task_id)
    await _record_dedup(effective_pool, key, session_id, brain_name, task_uuid)
    return WakeHandle(task_id=str(task_uuid), dedup_key=key, deduplicated=False)


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
    """Cancel any active wake tasks for this session+brain so `supersede` can replace them."""
    async with pool.connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        await cur.execute(
            'SELECT task_id FROM wake_dedup WHERE session_id = %s AND brain_name = %s',
            (session_id, brain_name),
        )
        rows = await cur.fetchall()
    for row in rows:
        await absurd.cancel_task(str(row['task_id']))
