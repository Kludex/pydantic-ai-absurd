from __future__ import annotations

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from .schema import MIGRATIONS_SQL


async def apply_migrations(pool: AsyncConnectionPool[AsyncConnection[tuple[object, ...]]]) -> None:
    """Apply the agent-sessions schema. Idempotent - safe to call on every startup."""
    async with pool.connection() as conn:
        await conn.execute(MIGRATIONS_SQL)
