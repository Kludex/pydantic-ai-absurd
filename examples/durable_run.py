"""Runnable version of the README example.

Spins up a Postgres testcontainer, installs the Absurd schema, registers an
agent as a durable `<name>.run` workflow, spawns a run, drains the worker, and
prints the result. A second spawn passes `message_history` to show the run
continuing the conversation.

Run with:

    OPENAI_API_KEY=... uv run python examples/durable_run.py
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from absurd_sdk import AsyncAbsurd, AsyncConnection
from pydantic_ai import Agent
from pydantic_ai_absurd import AbsurdAgent
from testcontainers.postgres import PostgresContainer

ABSURD_SQL = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "pydantic-ai-absurd"
    / "tests"
    / "fixtures"
    / "absurd.sql"
).read_text()


def _normalize_dsn(url: str) -> str:
    return url.replace("postgresql+psycopg2://", "postgresql://")


async def main() -> None:
    if "DOCKER_HOST" not in os.environ:
        home_sock = Path.home() / ".docker" / "run" / "docker.sock"
        if home_sock.exists():
            os.environ["DOCKER_HOST"] = f"unix://{home_sock}"

    with PostgresContainer("postgres:16-alpine") as container:
        dsn = _normalize_dsn(container.get_connection_url())
        with psycopg.connect(dsn, autocommit=True) as cx:
            cx.execute(ABSURD_SQL)

        async with await AsyncConnection.connect(dsn, autocommit=True) as conn:
            absurd = AsyncAbsurd(conn, queue_name="agents")
            await absurd.create_queue()

            agent = AbsurdAgent(
                Agent("openai:gpt-5.2", name="analyst"),
                absurd,
                name="analyst",
                register_task=True,
            )
            assert agent.task_name == "analyst.run"

            spawned = await absurd.spawn(
                "analyst.run", {"prompt": "In one sentence, what is durable execution?"}
            )
            await absurd.work_batch(batch_size=1)
            first = await absurd.fetch_task_result(spawned["task_id"])
            assert first is not None and first.state == "completed"
            print("first run output:", first.result["output"])

            # Continue the conversation: feed the prior messages back in.
            followup = await absurd.spawn(
                "analyst.run",
                {
                    "prompt": "Now say it again, but as a haiku.",
                    "message_history": first.result["all_messages"],
                },
            )
            await absurd.work_batch(batch_size=1)
            second = await absurd.fetch_task_result(followup["task_id"])
            assert second is not None and second.state == "completed"
            print("followup output:", second.result["output"])


if __name__ == "__main__":
    import anyio

    anyio.run(main)
