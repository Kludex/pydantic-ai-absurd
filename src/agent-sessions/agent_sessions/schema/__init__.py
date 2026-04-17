from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SCHEMA_DIR: Path = Path(__file__).resolve().parent
CANONICAL_SQL: str = (SCHEMA_DIR / 'agent_sessions.sql').read_text()
MIGRATIONS_DIR: Path = SCHEMA_DIR / 'migrations'

# Matches e.g. "0.0.1-0.0.2.sql" or "0.0.2-main.sql".
_MIGRATION_FILENAME_RE = re.compile(r'^(?P<from>[0-9]+\.[0-9]+\.[0-9]+)-(?P<to>main|[0-9]+\.[0-9]+\.[0-9]+)\.sql$')
_SCHEMA_VERSION_RE = re.compile(r"SELECT\s+'(?P<version>[^']+)'", re.IGNORECASE)


@dataclass(frozen=True)
class Migration:
    from_version: str
    to_version: str
    sql: str


def _parse_canonical_version() -> str:
    """Extract the current library version from `get_schema_version()` in agent_sessions.sql."""
    marker = 'CREATE OR REPLACE FUNCTION agent_sessions.get_schema_version'
    tail = CANONICAL_SQL.split(marker, 1)[-1]
    match = _SCHEMA_VERSION_RE.search(tail)
    if match is None:  # pragma: no cover - defensive; the canonical file ships with a version
        raise RuntimeError('agent_sessions.sql is missing a version marker')
    return match.group('version')


def load_migrations() -> list[Migration]:
    """Load all shipped migrations, sorted in application order.

    We assume strictly linear history: each migration's `to` matches the next
    one's `from`. If someone ships a branching history, startup will fail with a
    clear error rather than silently applying the wrong path.
    """
    if not MIGRATIONS_DIR.exists():
        return []
    by_from: dict[str, Migration] = {}
    for path in sorted(MIGRATIONS_DIR.iterdir()):
        if not path.is_file() or not path.name.endswith('.sql'):
            continue
        match = _MIGRATION_FILENAME_RE.match(path.name)
        if match is None:
            raise RuntimeError(f'migration file {path.name!r} does not match <from>-<to>.sql')
        m = Migration(
            from_version=match.group('from'),
            to_version=match.group('to'),
            sql=path.read_text(),
        )
        if m.from_version in by_from:
            raise RuntimeError(
                f'duplicate migration source {m.from_version!r}: {by_from[m.from_version].to_version} vs {m.to_version}'
            )
        by_from[m.from_version] = m
    return list(by_from.values())


CURRENT_VERSION: str = _parse_canonical_version()

__all__ = [
    'CANONICAL_SQL',
    'CURRENT_VERSION',
    'Migration',
    'load_migrations',
]
