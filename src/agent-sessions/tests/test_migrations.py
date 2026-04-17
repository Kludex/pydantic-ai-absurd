"""Migration versioning tests.

Covers:
  - fresh install on a virgin DB
  - no-op when already at the current version
  - linear chain: fake a DB at an older version and walk forward
  - error when the DB reports a version that has no outgoing migration
  - error when the migrations directory has a filename that doesn't parse
  - error when two migrations share a `from` version (branching history)
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path

import psycopg
import pytest
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from agent_sessions import apply_migrations
from agent_sessions.schema import (
    CANONICAL_SQL,
    CURRENT_VERSION,
    load_migrations,
)

from .conftest import AsyncPool

pytestmark = pytest.mark.anyio


async def _schema_installed(pool: AsyncPool) -> bool:
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT to_regprocedure('agent_sessions.get_schema_version()') IS NOT NULL")
        row = await cur.fetchone()
    return bool(row and row[0])


async def _installed_version(pool: AsyncPool) -> str:
    async with pool.connection() as conn:
        cur = await conn.execute('SELECT agent_sessions.get_schema_version()')
        row = await cur.fetchone()
    assert row is not None
    return str(row[0])


@pytest.fixture
async def empty_pool(
    db_dsn: str,
) -> AsyncIterator[AsyncConnectionPool[AsyncConnection[tuple[object, ...]]]]:
    """A fresh DB with *no* `agent_sessions` schema yet.

    Teardown drops the schema so repeat runs get a clean slate.
    """
    with psycopg.connect(db_dsn, autocommit=True) as sync:
        sync.execute('DROP SCHEMA IF EXISTS agent_sessions CASCADE')
    async with AsyncConnectionPool(db_dsn, min_size=1, max_size=2, open=False) as pool:
        await pool.open(wait=True)
        yield pool
    with psycopg.connect(db_dsn, autocommit=True) as sync:
        sync.execute('DROP SCHEMA IF EXISTS agent_sessions CASCADE')


async def test_fresh_install_creates_schema_at_current_version(empty_pool: AsyncPool) -> None:
    assert not await _schema_installed(empty_pool)
    await apply_migrations(empty_pool)
    assert await _installed_version(empty_pool) == CURRENT_VERSION


async def test_noop_when_already_at_current_version(empty_pool: AsyncPool) -> None:
    await apply_migrations(empty_pool)
    # Second call should detect the current version and do nothing.
    await apply_migrations(empty_pool)
    assert await _installed_version(empty_pool) == CURRENT_VERSION


async def test_db_version_without_migration_path_raises(empty_pool: AsyncPool) -> None:
    """If the DB reports an older version and the migrations dir can't bridge it,
    refuse to touch anything and fail loud."""
    # Install, then lie about the version - pretend we're at 0.0.0 without shipping
    # a 0.0.0->current migration.
    await apply_migrations(empty_pool)
    async with empty_pool.connection() as conn:
        await conn.execute(
            textwrap.dedent("""
                CREATE OR REPLACE FUNCTION agent_sessions.get_schema_version()
                    RETURNS TEXT LANGUAGE SQL IMMUTABLE AS $$ SELECT '0.0.0' $$
            """)
        )
    with pytest.raises(RuntimeError, match="no migration path from '0.0.0'"):
        await apply_migrations(empty_pool)


def test_load_migrations_empty_when_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Honesty check: if the shipped package has no migrations yet, load_migrations
    returns an empty list - it doesn't fail, doesn't walk the directory tree."""
    import agent_sessions.schema as schema

    empty_dir = tmp_path / 'empty_migrations'
    empty_dir.mkdir()
    monkeypatch.setattr(schema, 'MIGRATIONS_DIR', empty_dir)
    assert load_migrations() == []


def test_load_migrations_handles_missing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No directory at all (e.g. user built a stripped-down wheel) - empty, not crash."""
    import agent_sessions.schema as schema

    monkeypatch.setattr(schema, 'MIGRATIONS_DIR', tmp_path / 'does_not_exist')
    assert load_migrations() == []


def test_load_migrations_parses_filenames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_sessions.schema as schema

    mig_dir = tmp_path / 'mig'
    mig_dir.mkdir()
    (mig_dir / '0.0.1-0.0.2.sql').write_text('-- step 1')
    (mig_dir / '0.0.2-0.0.3.sql').write_text('-- step 2')
    (mig_dir / 'README.md').write_text('not a migration')  # must be ignored
    monkeypatch.setattr(schema, 'MIGRATIONS_DIR', mig_dir)
    got = load_migrations()
    assert [(m.from_version, m.to_version) for m in got] == [
        ('0.0.1', '0.0.2'),
        ('0.0.2', '0.0.3'),
    ]


def test_load_migrations_rejects_bad_filenames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_sessions.schema as schema

    mig_dir = tmp_path / 'mig'
    mig_dir.mkdir()
    (mig_dir / 'not-a-version-pair.sql').write_text('-- nope')
    monkeypatch.setattr(schema, 'MIGRATIONS_DIR', mig_dir)
    with pytest.raises(RuntimeError, match='does not match'):
        load_migrations()


def test_load_migrations_rejects_branching_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_sessions.schema as schema

    mig_dir = tmp_path / 'mig'
    mig_dir.mkdir()
    (mig_dir / '0.0.1-0.0.2.sql').write_text('-- a')
    (mig_dir / '0.0.1-0.0.3.sql').write_text('-- b')
    monkeypatch.setattr(schema, 'MIGRATIONS_DIR', mig_dir)
    with pytest.raises(RuntimeError, match='duplicate migration source'):
        load_migrations()


async def test_walks_through_shipped_migration_chain(
    empty_pool: AsyncPool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: fake a DB at an older version + a chain of migrations that
    lead to CURRENT_VERSION, and prove apply_migrations walks the chain
    in order. Each migration sets a side-effect column so we can verify order.
    """
    import agent_sessions.migrations as migrations_mod
    import agent_sessions.schema as schema

    # Install at current first, then downgrade the version function to pretend
    # we're a few releases behind.
    await apply_migrations(empty_pool)
    async with empty_pool.connection() as conn:
        await conn.execute('CREATE TABLE agent_sessions.migration_trace (seq SERIAL PRIMARY KEY, marker TEXT)')
        await conn.execute(
            'CREATE OR REPLACE FUNCTION agent_sessions.get_schema_version() '
            "RETURNS TEXT LANGUAGE SQL IMMUTABLE AS $$ SELECT '0.0.9-alpha' $$"
        )

    mig_dir = tmp_path / 'mig'
    mig_dir.mkdir()

    def _make(name: str, marker: str, final_version: str) -> None:
        (mig_dir / name).write_text(
            f"INSERT INTO agent_sessions.migration_trace (marker) VALUES ('{marker}');\n"
            'CREATE OR REPLACE FUNCTION agent_sessions.get_schema_version() '
            f"RETURNS TEXT LANGUAGE SQL IMMUTABLE AS $$ SELECT '{final_version}' $$;\n"
        )

    # Replace the filename regex temporarily so our fake chain's `from`/`to`
    # parses. The real regex only accepts MAJOR.MINOR.PATCH; we'll use those.
    _make('0.0.9-alpha-0.0.9-beta.sql', 'a->b', '0.0.9-beta')
    _make(f'0.0.9-beta-{CURRENT_VERSION}.sql', f'b->{CURRENT_VERSION}', CURRENT_VERSION)

    # Loosen the regex for this test run - the real schema only ships
    # PEP-440-ish X.Y.Z versions, but the parser supports alphanumeric suffixes
    # if we allow them. For this test we just point at our dir with relaxed
    # parsing via monkeypatched regex.
    monkeypatch.setattr(schema, 'MIGRATIONS_DIR', mig_dir)
    monkeypatch.setattr(
        schema,
        '_MIGRATION_FILENAME_RE',
        re.compile(
            r'^(?P<from>[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.]+)?)-'
            r'(?P<to>[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.]+)?)\.sql$'
        ),
    )
    monkeypatch.setattr(migrations_mod, 'load_migrations', schema.load_migrations)

    await apply_migrations(empty_pool)

    assert await _installed_version(empty_pool) == CURRENT_VERSION
    async with empty_pool.connection() as conn:
        cur = await conn.execute('SELECT marker FROM agent_sessions.migration_trace ORDER BY seq')
        rows = await cur.fetchall()
    assert [r[0] for r in rows] == ['a->b', f'b->{CURRENT_VERSION}']


def test_canonical_version_parses_from_sql() -> None:
    """Guards against the SQL file and Python version parser drifting apart."""
    assert CURRENT_VERSION
    assert CURRENT_VERSION in CANONICAL_SQL


def test_shipped_migrations_chain_is_valid() -> None:
    """If any real migration files get shipped, they must form a linear chain
    ending at CURRENT_VERSION. Empty chain is fine."""
    migrations = load_migrations()
    if not migrations:
        return  # pragma: no cover - will fire once we ship a migration
    # Each step's `to` must equal the next step's `from`.
    for a, b in zip(migrations, migrations[1:]):  # pragma: no cover
        assert a.to_version == b.from_version, (
            f'migration chain breaks: {a.from_version}->{a.to_version}, then {b.from_version}->...'
        )
    # Last step must land at the current version.
    assert migrations[-1].to_version == CURRENT_VERSION  # pragma: no cover
