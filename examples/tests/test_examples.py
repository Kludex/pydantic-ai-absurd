from __future__ import annotations

import pytest

pytestmark = pytest.mark.anyio


async def test_durable_run_example() -> None:
    import durable_run

    await durable_run.main()
