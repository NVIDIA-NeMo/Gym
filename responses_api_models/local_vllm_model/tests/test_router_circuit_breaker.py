# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned, patched router against a real HTTP worker; no GPU required."""

import hashlib
import os
import socket
from pathlib import Path

import pytest
from aiohttp import web

from nemo_gym import server_utils
from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig, request, set_global_aiohttp_client
from responses_api_models.local_vllm_model.router_launcher import VLLMRouterConfig, VLLMRouterLauncher


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["consistent_hash", "cache_aware"])
@pytest.mark.parametrize("status", [200, 429, 500, "disconnect"])
async def test_transparent_outcomes_without_replaying_generation(tmp_path, policy, status):
    executable = os.getenv("VLLM_ROUTER_TEST_EXECUTABLE")
    if not executable:
        pytest.skip("Set VLLM_ROUTER_TEST_EXECUTABLE to the pinned binary built with router_compat.patch")
    calls = []

    async def worker(req):
        if req.path == "/health":
            return web.json_response({"healthy": True})
        if req.path == "/v1/models":
            return web.json_response({"data": [{"id": "test-model"}]})
        calls.append((req.path, await req.json()))
        if status == "disconnect":
            req.transport.close()
            return web.Response()
        return web.json_response({"test_status": status}, status=status)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", worker)
    worker_runner = web.AppRunner(app)
    await worker_runner.setup()
    site = web.TCPSite(worker_runner, "127.0.0.1", 0)
    await site.start()
    worker_port = site._server.sockets[0].getsockname()[1]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        router_port = sock.getsockname()[1]
    client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    router = VLLMRouterLauncher(
        config=VLLMRouterConfig(
            executable=executable,
            executable_type="binary",
            expected_sha256=hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
            log_dir=tmp_path,
            policy=policy,
            startup_timeout_seconds=20,
            shutdown_timeout_seconds=1,
            inference_timeout_seconds=3,
        ),
        model="test-model",
        api_key="test-key",
    )
    try:
        base = await router.start(router_port, worker_urls=[f"http://127.0.0.1:{worker_port}"])
        expected = 502 if status == "disconnect" else status
        for index, path in enumerate(("/chat/completions", "/completions")):
            response = await request(
                "POST",
                base + path,
                json={"model": "test-model", "prompt": "hello"},
                headers={"Authorization": "Bearer test-key", "X-Session-ID": "episode"},
                _max_connection_retries=0,
            )
            async with response:
                await response.read()
                assert response.status == (503 if index and status in (500, "disconnect") else expected)
        # Health is still good, but a failed generation must open the circuit.
        response = await request("GET", f"http://127.0.0.1:{worker_port}/health")
        async with response:
            assert response.status == 200
        assert len(calls) == (1 if status in (500, "disconnect") else 2)
    finally:
        await router.stop()
        await client.close()
        server_utils._GLOBAL_AIOHTTP_CLIENT = None
        await worker_runner.cleanup()
