# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concurrent real SimpleAgent runs, including self-calls, tools and verification."""

import asyncio
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from omegaconf import OmegaConf

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.model_endpoint import current_model_endpoint
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


def test_concurrent_native_agents_keep_credentials_out_of_tools_and_verifiers(monkeypatch):
    monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", "{}")

    async def run():
        client = ServerClient(
            head_server_config=BaseServerConfig(host="head", port=1),
            global_config_dict=OmegaConf.create(
                {
                    "agent": {"responses_api_agents": {}},
                    "resources": {"resources_servers": {}},
                    "policy": {"responses_api_models": {}},
                }
            ),
        )
        client._server_base_urls.update({"agent": "http://agent", "resources": "http://resources"})
        agent = SimpleAgent(
            config=SimpleAgentConfig(
                host="agent",
                port=1,
                entrypoint="",
                name="agent",
                model_server={"type": "responses_api_models", "name": "policy"},
                resources_server={"type": "resources_servers", "name": "resources"},
            ),
            server_client=client,
        )
        model, resources = FastAPI(), FastAPI()
        seen = []
        first_keys = set()
        both_started = asyncio.Event()

        @model.post("/v1/responses")
        async def respond(request: Request):
            body = await request.json()
            key = request.headers["authorization"].removeprefix("Bearer ")
            expected = body["input"][0]["content"]
            assert key == expected
            seen.append((key, "model"))
            first_keys.add(key)
            if len(first_keys) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), 5)
            tool_done = any(item.get("type") == "function_call_output" for item in body["input"])
            output = (
                [
                    {
                        "type": "message",
                        "id": "msg",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "done", "annotations": []}],
                    }
                ]
                if tool_done
                else [
                    {
                        "type": "function_call",
                        "id": "fc",
                        "call_id": "call",
                        "name": "echo",
                        "arguments": '{"value":"ok"}',
                        "status": "completed",
                    }
                ]
            )
            return {
                "id": "resp",
                "object": "response",
                "created_at": 0,
                "model": "policy",
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }

        @resources.post("/{operation}")
        async def resource(operation: str, request: Request):
            assert "authorization" not in request.headers
            assert not any(key.startswith("x-gym-model") for key in request.headers)
            body = await request.json()
            assert "model_endpoint" not in body
            seen.append((None, operation))
            if operation == "verify":
                assert body["response"]["output"][-1]["content"][0]["text"] == "done"
                return {**body, "reward": 1.0}
            return {"value": "ok"}

        apps = {"agent": agent.setup_webserver(), "resources": resources, "model": model}

        class Response:
            def __init__(self, response):
                self.status = response.status_code
                self.ok = self.status < 400
                self.cookies = {}
                self.content = self
                self._response = response

            async def read(self):
                return self._response.content

        async def request(method, url, **kwargs):
            kwargs.pop("_internal", None)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=apps[urlsplit(url).hostname])) as http:
                return Response(await http.request(method, url, **kwargs))

        monkeypatch.setattr("nemo_gym.server_utils.request", request)

        async def rollout(key):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=apps["agent"]), base_url="http://agent"
            ) as http:
                response = await http.post(
                    "/run",
                    json={
                        "responses_create_params": {"input": [{"role": "user", "content": key}]},
                        "model_endpoint": {"server_name": "policy", "base_url": "http://model/v1", "api_key": key},
                    },
                )
                assert response.status_code == 200, response.text
                result = response.json()
                assert result["reward"] == 1
                assert "model_endpoint" not in result

        await asyncio.gather(rollout("one"), rollout("two"))
        assert seen.count(("one", "model")) == seen.count(("two", "model")) == 2
        assert seen.count((None, "verify")) == seen.count((None, "echo")) == 2
        assert current_model_endpoint() is None

    asyncio.run(run())
