from __future__ import annotations

from pathlib import Path

MIGRATIONS_SQL: str = (Path(__file__).resolve().parent / 'migrations.sql').read_text()

__all__ = ['MIGRATIONS_SQL']
