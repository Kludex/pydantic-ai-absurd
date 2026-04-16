from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from absurd_sdk import AsyncAbsurd
from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool
from testcontainers.postgres import PostgresContainer

from agent_sessions import Workflow, apply_migrations

FIXTURES = Path(__file__).resolve().parent.parent.parent / 'pydantic-ai-absurd' / 'tests' / 'fixtures'
ABSURD_SQL = (FIXTURES / 'absurd.sql').read_text()

AsyncPool = AsyncConnectionPool[AsyncConnection[TupleRow]]


def _docker_host_env() -> None:  # pragma: no cover - environment-dependent
    if 'DOCKER_HOST' in os.environ:
        return
    home_sock = Path.home() / '.docker' / 'run' / 'docker.sock'
    if home_sock.exists():
        os.environ['DOCKER_HOST'] = f'unix://{home_sock}'


def _normalize_dsn(url: str) -> str:
    if url.startswith('postgresql+psycopg2://'):
        return 'postgresql://' + url.split('://', 1)[1]
    return url  # pragma: no cover


@pytest.fixture(scope='session')
def postgres_container() -> Iterator[PostgresContainer]:
    _docker_host_env()
    container = PostgresContainer('postgres:16-alpine')
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope='session')
def db_dsn(postgres_container: PostgresContainer) -> str:
    dsn = _normalize_dsn(postgres_container.get_connection_url())
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(ABSURD_SQL)
    return dsn


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
async def pool(db_dsn: str) -> AsyncIterator[AsyncPool]:
    async with AsyncConnectionPool(db_dsn, min_size=1, max_size=4, open=False) as pool:
        await pool.open(wait=True)
        await apply_migrations(pool)
        yield pool


@pytest.fixture
async def absurd(db_dsn: str) -> AsyncIterator[AsyncAbsurd]:
    queue = f'test_{uuid4().hex[:8]}'
    async with await AsyncConnection.connect(db_dsn, autocommit=True) as conn:
        client = AsyncAbsurd(conn, queue_name=queue)
        await client.create_queue()
        try:
            yield client
        finally:
            await client.drop_queue()


@pytest.fixture
def workflow(pool: AsyncPool, absurd: AsyncAbsurd) -> Workflow:
    return Workflow(absurd=absurd, pool=pool)
