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

ABSURD_SQL = (Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "absurd.sql").read_text()


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

            agent = AbsurdAgent(Agent("openai:gpt-5.2", name="analyst"), absurd)

            # Author the task; call agent.run() inside it. Each model/MCP call is
            # checkpointed, so a crash mid-run resumes from the last completed step.
            @absurd.register_task(name="analyse")
            async def analyse(params, ctx):
                result = await agent.run(params["prompt"])
                return {"output": result.output}

            spawned = await absurd.spawn(
                "analyse", {"prompt": "In one sentence, what is durable execution?"}
            )
            await absurd.work_batch(batch_size=1)
            done = await absurd.fetch_task_result(spawned["task_id"])
            assert done is not None and done.state == "completed"
            print("output:", done.result["output"])


if __name__ == "__main__":
    import anyio

    anyio.run(main)
