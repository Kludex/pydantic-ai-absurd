"""Tiny FastMCP server used by tests via MCPServerStdio."""

from __future__ import annotations

from fastmcp import FastMCP

server: FastMCP[None] = FastMCP(name='tiny')


@server.tool
def add(a: int, b: int) -> int:
    return a + b


if __name__ == '__main__':
    server.run()
