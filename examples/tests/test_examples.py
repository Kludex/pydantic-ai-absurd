from __future__ import annotations

import pytest

pytestmark = pytest.mark.anyio


async def test_readme_snippet_runs() -> None:
    import readme_snippet

    await readme_snippet.main()


async def test_vercel_starlette_runs() -> None:
    import vercel_starlette

    await vercel_starlette.main()
