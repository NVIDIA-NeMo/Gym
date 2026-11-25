# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the staged standard-library bridge against a real local HTTP server."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp import web

from resources_servers.mcp_atlas import sandbox_http


@pytest.fixture
async def http_server():
    calls = []
    release = asyncio.Event()
    state = {"status": 200, "body": b'{"ok":true}', "headers": {}}

    async def handle(request: web.Request) -> web.Response:
        calls.append((request.method, request.path, await request.read()))
        if state.get("disconnect"):
            request.transport.abort()
        if state.get("wait"):
            await release.wait()
        return web.Response(status=state["status"], body=state["body"], headers=state["headers"])

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield port, calls, state
    finally:
        release.set()
        await runner.cleanup()


async def run_helper(port: int, request: object = None, *, raw: str | None = None) -> dict:
    # Change only the external server port. Run the actual staged source, with no Gym imports/dependencies.
    command = "import runpy,sys; runpy.run_path(sys.argv[1])['main'](port=int(sys.argv[2]))"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        command,
        str(Path(sandbox_http.__file__)),
        str(port),
        env={
            sandbox_http.REQUEST_ENV: raw if raw is not None else json.dumps(request),
            "HTTP_PROXY": "http://127.0.0.1:1",
            "ALL_PROXY": "http://127.0.0.1:1",
        },
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, stderr.decode(errors="replace")
    assert not stderr
    return json.loads(stdout.decode(errors="replace"))


def bridge_request(*, method="POST", path="/call-tool", timeout_s=2, payload=None) -> dict:
    return {"method": method, "path": path, "timeout_s": timeout_s, "json": payload}


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (200, b'{"ok":true}', {"ok": True}),
        (500, b'{"detail":"tool failed"}', {"detail": "tool failed"}),
        (429, b"rate limited", "rate limited"),
        (204, b"", ""),
        (200, b"bad \xff byte", "bad \ufffd byte"),
    ],
)
async def test_bridge_preserves_http_status_and_json_or_text(http_server, status, body, expected) -> None:
    port, calls, state = http_server
    state.update(status=status, body=body)
    assert await run_helper(port, bridge_request(payload={"tool_args": {"x": 1}})) == {
        "status": status,
        "body": expected,
    }
    assert len(calls) == 1
    assert calls[0][:2] == ("POST", "/call-tool")
    assert json.loads(calls[0][2]) == {"tool_args": {"x": 1}}


@pytest.mark.parametrize(("method", "path"), [("GET", "/health"), ("POST", "/list-tools")])
async def test_bridge_supports_readiness_routes(http_server, method, path) -> None:
    port, calls, _ = http_server
    assert (await run_helper(port, bridge_request(method=method, path=path)))["status"] == 200
    assert calls == [(method, path, b"")]


async def test_request_arguments_are_data_and_do_not_execute_shell_syntax(http_server, tmp_path) -> None:
    port, calls, _ = http_server
    marker = tmp_path / "injected"
    arguments = {"query": f"'; touch {marker}; $(touch {marker}) `touch {marker}`\n$HOME", "nested": ["\\", '"']}
    payload = {"tool_name": "fetch", "tool_args": arguments, "use_cache": False}
    assert (await run_helper(port, bridge_request(payload=payload)))["status"] == 200
    assert json.loads(calls[0][2]) == payload
    assert not marker.exists()


@pytest.mark.parametrize(
    "override",
    [
        {"method": "DELETE"},
        {"path": "http://example.invalid/control"},
        {"path": "//example.invalid/health"},
        {"path": "/health?url=http://example.invalid"},
        {"path": "/../control"},
        {"host": "example.invalid"},
        {"port": 9999},
        {"timeout_s": 0},
        {"timeout_s": True},
        {"timeout_s": float("inf")},
        {"timeout_s": "2"},
        {"method": []},
    ],
)
async def test_invalid_requests_cannot_change_destination_or_reach_the_server(http_server, override) -> None:
    port, calls, _ = http_server
    result = await run_helper(port, {**bridge_request(), **override})
    assert result["error"] == "invalid_request"
    assert calls == []


@pytest.mark.parametrize("raw", ["not JSON", "[]", "null", "{}"])
async def test_malformed_request_has_an_explicit_error_without_http(http_server, raw) -> None:
    port, calls, _ = http_server
    assert (await run_helper(port, raw=raw))["error"] == "invalid_request"
    assert calls == []


async def test_http_redirect_is_returned_without_following_it(http_server) -> None:
    port, calls, state = http_server
    state.update(status=302, body=b"redirect", headers={"Location": f"http://127.0.0.1:{port}/control"})
    assert await run_helper(port, bridge_request()) == {"status": 302, "body": "redirect"}
    assert len(calls) == 1


async def test_timeout_is_explicit_and_does_not_replay_a_mutating_request(http_server) -> None:
    port, calls, state = http_server
    state["wait"] = True
    assert (await run_helper(port, bridge_request(timeout_s=0.05)))["error"] == "timeout"
    assert len(calls) == 1


async def test_disconnected_response_is_explicit_and_not_replayed(http_server) -> None:
    port, calls, state = http_server
    state["disconnect"] = True
    assert (await run_helper(port, bridge_request()))["error"] == "transport"
    assert len(calls) == 1
