from __future__ import annotations

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from .schema import CANONICAL_SQL, CURRENT_VERSION, Migration, load_migrations


async def apply_migrations(
    pool: AsyncConnectionPool[AsyncConnection[tuple[object, ...]]],
) -> None:
    """Bring the database's `agent_sessions` schema up to the library's version.

    Fresh install: the `agent_sessions.get_schema_version()` function doesn't
    exist yet, so we run the canonical `agent_sessions.sql` which creates every
    table and registers the version function.

    Existing install: we read the current version from
    `agent_sessions.get_schema_version()`, then apply the deltas under
    `schema/migrations/<from>-<to>.sql` in order until we reach
    `CURRENT_VERSION`. Each migration is applied in its own transaction so a
    failure leaves the database at a known intermediate version.

    Safe to call on every startup - idempotent when already at the current
    version. Must not be called concurrently from multiple processes; use
    your deployment's "run once" hook.
    """
    installed = await _read_installed_version(pool)
    if installed is None:
        await _install_fresh(pool)
        return
    if installed == CURRENT_VERSION:
        return
    await _apply_chain(pool, installed)


async def _read_installed_version(
    pool: AsyncConnectionPool[AsyncConnection[tuple[object, ...]]],
) -> str | None:
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT to_regprocedure('agent_sessions.get_schema_version()') IS NOT NULL")
        row = await cur.fetchone()
        if row is None or not row[0]:
            return None
        cur = await conn.execute('SELECT agent_sessions.get_schema_version()')
        version_row = await cur.fetchone()
    if version_row is None:
        return None  # pragma: no cover - function existed at check time
    return str(version_row[0])


async def _install_fresh(
    pool: AsyncConnectionPool[AsyncConnection[tuple[object, ...]]],
) -> None:
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute(CANONICAL_SQL)


async def _apply_chain(
    pool: AsyncConnectionPool[AsyncConnection[tuple[object, ...]]],
    installed: str,
) -> None:
    migrations = {m.from_version: m for m in load_migrations()}
    current = installed
    applied: list[Migration] = []
    seen: set[str] = set()
    while current != CURRENT_VERSION:
        if current in seen:  # pragma: no cover - defensive cycle detection
            raise RuntimeError(f'migration cycle detected at {current!r}')
        seen.add(current)
        step = migrations.get(current)
        if step is None:
            raise RuntimeError(
                f'database reports schema version {installed!r} but no migration path '
                f'from {current!r} to {CURRENT_VERSION!r} is shipped with this release'
            )
        async with pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(step.sql)
        applied.append(step)
        current = step.to_version
