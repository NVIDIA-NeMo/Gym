# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise agent mutations through ServerClient and real aiohttp transport."""

import asyncio
from collections import Counter

import aiohttp
import pytest
from aiohttp import web
from omegaconf import OmegaConf

from nemo_gym.client import ServerClient, get_global_aiohttp_client
from nemo_gym.config import BaseServerConfig
from responses_api_agents.visgym_agent.app import TextActionAgentRunRequest
from responses_api_agents.visgym_agent.tests.test_app import _make_agent, _model_response, _seed_session, _step


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_seed", [False, True])
@pytest.mark.parametrize("observability", [False, True])
async def test_agent_http_mutations_are_bounded(drop_seed, observability):
    model_path = "/ng-rollout/7-2-a1/v1/responses" if observability else "/v1/responses"
    request_body = TextActionAgentRunRequest(task_idx=0, _ng_task_index=7, _ng_rollout_index=2, _ng_attempt_index=1)
    calls = Counter()
    payloads = []

    async def handle(request):
        body = await request.json()
        calls[request.path] += 1
        payloads.append((request.path, body))
        if request.path == "/seed_session":
            if drop_seed:
                request.transport.close()
                return web.Response()
            return web.json_response(_seed_session(env_id=body["env_id"]))
        if request.path == model_path:
            response = _model_response(r"\boxed{step}")
            response["usage"] = {
                "input_tokens": 10 * calls[model_path],
                "output_tokens": 3 * calls[model_path],
                "total_tokens": 13 * calls[model_path],
                "input_tokens_details": {"cached_tokens": calls[model_path]},
                "output_tokens_details": {"reasoning_tokens": calls[model_path]},
            }
            return web.json_response(response)
        if request.path == "/step":
            return web.json_response(_step(done=calls["/step"] == 2, reward=1.0))
        if request.path == "/close":
            # Cleanup is idempotent and may retry; mutations must not replay.
            if calls[request.path] < 3:
                request.transport.close()
                return web.Response()
            return web.json_response({"success": True})
        if request.path == "/verify":
            return web.json_response({**body, "reward": 1.0, "task_success": True})
        raise AssertionError(request.path)

    app = web.Application()
    app.router.add_post("/{path:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=port),
        global_config_dict=OmegaConf.create(
            {
                "observability_enabled": observability,
                "my resources name": {"resources_servers": {"test": {"host": "127.0.0.1", "port": port}}},
                "my model name": {"responses_api_models": {"test": {"host": "127.0.0.1", "port": port}}},
            }
        ),
    )
    agent = _make_agent(max_steps=2)
    agent.server_client = client
    try:
        async with asyncio.timeout(10):
            if drop_seed:
                with pytest.raises(aiohttp.ServerDisconnectedError):
                    await agent.run(request_body)
                assert calls == {"/seed_session": 1, "/close": 3}
                assert payloads[-1][1]["discard_reward"] is True
            else:
                result = await agent.run(request_body)
                assert result.reward == 1.0
                assert result.task_success is True
                assert result.response.usage.input_tokens == 30
                assert result.response.usage.output_tokens == 9
                assert result.response.usage.total_tokens == 39
                assert result.response.usage.input_tokens_details.cached_tokens == 3
                assert result.response.usage.output_tokens_details.reasoning_tokens == 3
                assert calls == {"/seed_session": 1, model_path: 2, "/step": 2, "/close": 3, "/verify": 1}
    finally:
        await get_global_aiohttp_client().close()
        await runner.cleanup()
