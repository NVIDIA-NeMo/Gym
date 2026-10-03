# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise cookie preservation through the agent's real HTTP call loop."""

import asyncio

import pytest
import uvicorn
from aiohttp import ClientSession, DummyCookieJar, web
from omegaconf import OmegaConf

import nemo_gym.server_utils as server_utils
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_reply", ["error", "empty_delta", "rotate"])
async def test_cookie_deltas_preserve_episode_session(monkeypatch, tool_reply):
    state = {"model_calls": 0, "tool_calls": 0, "verify_calls": 0, "provider": "seeded"}

    async def seed(request):
        response = web.json_response({})
        response.set_cookie("provider_session", state["provider"])
        return response

    async def tool(request):
        assert request.cookies.get("provider_session") == state["provider"]
        state["tool_calls"] += 1
        if tool_reply == "rotate":
            state["provider"] = "rotated"
        response = web.json_response(
            {"output": "tool error" if tool_reply == "error" else "tool output"},
            status=500 if tool_reply == "error" else 200,
        )
        if tool_reply == "rotate":
            response.set_cookie("provider_session", state["provider"])
        return response

    async def verify(request):
        state["verify_calls"] += 1
        if request.cookies.get("provider_session") != state["provider"]:
            return web.json_response({"detail": "provider session is not active"}, status=409)
        return web.json_response((await request.json()) | {"reward": 0.0})

    async def model(request):
        state["model_calls"] += 1
        number = state["model_calls"]
        if number > 1:
            assert request.cookies.get("model_session") == "active"
        body = await request.json()
        if number == 1:
            output = [
                {
                    "id": "fc",
                    "call_id": "call",
                    "name": "tool",
                    "arguments": "{}",
                    "type": "function_call",
                    "status": "completed",
                }
            ]
        else:
            assert any(item.get("type") == "function_call_output" for item in body["input"])
            if tool_reply == "error":
                assert "tool error" in next(
                    item["output"] for item in body["input"] if item.get("type") == "function_call_output"
                )
            output = [
                {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}],
                }
            ]
        response = web.json_response(
            {
                "id": f"response-{number}",
                "created_at": float(number),
                "model": "policy",
                "object": "response",
                "status": "completed",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "output": output,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )
        if number == 1:
            response.set_cookie("model_session", "active")
        return response

    upstream = web.Application()
    upstream.router.add_post("/seed_session", seed)
    upstream.router.add_post("/tool", tool)
    upstream.router.add_post("/verify", verify)
    upstream.router.add_post("/v1/responses", model)
    runner = web.AppRunner(upstream)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    upstream_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    session = ClientSession(cookie_jar=DummyCookieJar())
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
        global_config_dict=OmegaConf.create({}),
    )
    client._server_base_urls.update({"model": upstream_url, "resources": upstream_url})
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="",
            name="simple",
            model_server=ModelServerRef(type="responses_api_models", name="model"),
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        ),
        server_client=client,
    )
    app = agent.setup_webserver()
    agent.setup_exception_middleware(app)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="off", log_level="error"))
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(500):
            if server.started and server.servers:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("agent HTTP server did not start")
        client._server_base_urls["simple"] = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
        async with ClientSession(cookie_jar=DummyCookieJar()) as caller:
            async with caller.post(
                client._server_base_urls["simple"] + "/run",
                json={"responses_create_params": {"input": [{"role": "user", "content": "task"}]}},
            ) as response:
                result = await response.json()
                assert response.status == 200, result
        assert result["reward"] == 0.0
        assert state["model_calls"] == 2
        assert state["tool_calls"] == 1
        assert state["verify_calls"] == 1
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        await session.close()
        await runner.cleanup()
