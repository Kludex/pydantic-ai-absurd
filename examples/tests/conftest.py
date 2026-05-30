from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the `examples` directory importable so tests can `import durable_run`.
EXAMPLES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLES_DIR))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip every test in this directory unless OPENAI_API_KEY is set."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    skip = pytest.mark.skip(
        reason="OPENAI_API_KEY not set; example tests hit the real API"
    )
    for item in items:
        item.add_marker(skip)
