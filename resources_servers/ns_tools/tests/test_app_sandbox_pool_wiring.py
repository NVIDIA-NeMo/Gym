# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app import NSToolsConfig, NSToolsResourcesServer
from nemo_skills.code_execution import sandbox as ns_sandbox

from nemo_gym.server_utils import ServerClient


class _FakePool:
    def __init__(self, **_config):
        self.start_calls = 0
        self.close_calls = 0

    async def start(self):
        self.start_calls += 1

    async def aclose(self):
        self.close_calls += 1


class _FakeToolManager:
    def __init__(self, module_specs, overrides, context):
        sandbox_config = dict(context["sandbox"])
        sandbox_type = sandbox_config.pop("sandbox_type")
        self.sandbox = ns_sandbox.get_sandbox(sandbox_type=sandbox_type, **sandbox_config)

    async def list_all_tools(self):
        return []

    async def shutdown(self):
        await self.sandbox.close()


def _server() -> NSToolsResourcesServer:
    config = NSToolsConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="ns_tools",
        nemo_skills_tools=["fake::DirectPythonTool"],
        sandbox_type="sandbox_pool",
        sandbox_pool={},
    )
    return NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_unknown_sandbox_type_is_rejected():
    with pytest.raises(ValueError, match="sandbox_type"):
        NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
            sandbox_type="sandbox_pol",
        )


def test_each_server_starts_and_closes_only_its_pool():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with (
            patch("app.ToolManager", _FakeToolManager),
            patch("sandbox_pool.SandboxPool", _FakePool),
            patch.object(NSToolsResourcesServer, "_tool_uses_python_tool_sidecar", return_value=False),
            patch.dict(ns_sandbox.sandboxes, {}, clear=True),
        ):
            first = _server()
            second = _server()
            first_app = first.setup_webserver()
            second_app = second.setup_webserver()
            from gym_sandbox import GymSandbox

            assert ns_sandbox.sandboxes["sandbox_pool"] is GymSandbox
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    first_pool = first._sandbox_pool
    second_pool = second._sandbox_pool
    assert first_pool is not second_pool
    assert first.tool_manager.sandbox._pool is first_pool
    assert second.tool_manager.sandbox._pool is second_pool

    async def run_lifespans():
        async with first_app.router.lifespan_context(first_app):
            async with second_app.router.lifespan_context(second_app):
                assert first_pool.start_calls == 1
                assert second_pool.start_calls == 1
                assert first_pool.close_calls == 0
                assert second_pool.close_calls == 0

            assert first_pool.close_calls == 0
            assert second_pool.close_calls == 1

    asyncio.run(run_lifespans())
    assert first_pool.close_calls == 1
    assert second_pool.close_calls == 1


@pytest.mark.parametrize("failure", [RuntimeError("failed"), asyncio.CancelledError()])
def test_shutdown_closes_pool_before_propagating_tool_manager_failure(failure):
    server = _server()
    server.tool_manager = MagicMock()
    server.tool_manager.shutdown = AsyncMock(side_effect=failure)
    server._sandbox_pool = _FakePool()

    with pytest.raises(type(failure)):
        asyncio.run(server.shutdown())

    assert server._sandbox_pool.close_calls == 1
