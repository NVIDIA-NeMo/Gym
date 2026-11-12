# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise MCP authentication rejection through the actual SimpleAgent and resources HTTP routes."""

import asyncio
from itertools import count
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner
from multidict import CIMultiDict, CIMultiDictProxy
from pydantic import BaseModel
from yarl import URL

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import EpisodeId
from nemo_gym.server_runtime import ResourcesCloseSessionRequest
from resources_servers.mcp_atlas import app as mcp_app
from resources_servers.mcp_atlas.app import MCPAtlasResourcesServer, MCPAtlasResourcesServerConfig
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


TOOL = "github_get_repository"


class _AiohttpReply:
    """Adapt an actual in-process HTTP response at the ServerClient transport boundary."""

    def __init__(self, response: httpx.Response) -> None:
        self.status = response.status_code
        self.ok = self.status < 400
        self.headers = CIMultiDict(response.headers)
        self.cookies = dict(response.cookies.items())
        url = URL(str(response.request.url))
        self.request_info = RequestInfo(
            url, response.request.method, CIMultiDictProxy(CIMultiDict(response.request.headers)), url
        )
        self._body = response.content
        self.content = SimpleNamespace(read=self.read)

    async def read(self) -> bytes:
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise ClientResponseError(
                self.request_info, (), status=self.status, message="synthetic ASGI reply", headers=self.headers
            )


def _json_body(body: object) -> object:
    return body.model_dump(mode="json") if isinstance(body, BaseModel) else body


def _model_reply(identifier: str, output: list[dict]) -> dict:
    return {
        "id": identifier,
        "created_at": 0.0,
        "model": "synthetic-policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def test_authentication_rejection_reaches_policy_and_keeps_run_ungraded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force renewed signatures so stale-cookie forwarding fails independently of second boundaries.
    timestamps = count(1_800_000_000)
    monkeypatch.setattr(TimestampSigner, "get_timestamp", lambda self: next(timestamps))
    judge = MagicMock(spec=ServerClient)
    judge.global_config_dict = {}
    judge.post = AsyncMock(side_effect=AssertionError("auth-rejected rollout must never call the judge"))
    resources = MCPAtlasResourcesServer(
        config=MCPAtlasResourcesServerConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            name="mcp_atlas",
            sandbox_base_url="http://synthetic-tool-transport.invalid",
            sandbox_development_mode=True,
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        ),
        server_client=judge,
    )
    transport = AsyncMock(return_value=(401, {"detail": "synthetic credential rejection"}))
    monkeypatch.setattr(mcp_app, "call_sandbox_tool", transport)
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            name="simple",
            model_server=ModelServerRef(type="responses_api_models", name="model"),
            resources_server=ResourcesServerRef(type="resources_servers", name="mcp_atlas"),
        ),
        server_client=client,
    )
    resources_app = resources.setup_webserver()
    resources.setup_exception_middleware(resources_app)
    agent_app = agent.setup_webserver()
    agent.setup_exception_middleware(agent_app)
    outputs = iter(
        [
            _model_reply(
                "synthetic-call",
                [
                    {
                        "id": "fc",
                        "type": "function_call",
                        "call_id": "call",
                        "name": TOOL,
                        "arguments": "{}",
                        "status": "completed",
                    }
                ],
            ),
            _model_reply(
                "synthetic-final",
                [
                    {
                        "id": "answer",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "synthetic final answer", "annotations": []}],
                    }
                ],
            ),
        ]
    )
    resource_calls: list[dict] = []
    model_inputs: list[dict] = []
    latest_cookies: dict[str, str] = {}
    with (
        TestClient(resources_app, raise_server_exceptions=False) as resource_http,
        TestClient(agent_app, raise_server_exceptions=False) as agent_http,
    ):

        async def post(*, server_name: str, url_path: str, **kwargs) -> _AiohttpReply:
            body = _json_body(kwargs.get("json"))
            if server_name == "model":
                model_inputs.append(body)
                return _AiohttpReply(
                    httpx.Response(200, json=next(outputs), request=httpx.Request("POST", "http://model/v1/responses"))
                )
            target = resource_http if server_name == "mcp_atlas" else agent_http
            assert server_name in {"mcp_atlas", "simple"}, "only actual resource/agent ASGI routes are bridged"
            if server_name == "mcp_atlas":
                # Require the agent's forwarded cookies; TestClient's own jar must not repair a lost cookie.
                resource_http.cookies.clear()
            reply = await asyncio.to_thread(
                target.post, url_path, json=body, cookies=kwargs.get("cookies"), headers=kwargs.get("headers")
            )
            if server_name == "mcp_atlas":
                resource_calls.append(
                    {
                        "path": url_path,
                        "status": reply.status_code,
                        "sent_cookie": bool(kwargs.get("cookies")),
                        "sent_latest_cookie": bool(latest_cookies)
                        and all(
                            (kwargs.get("cookies") or {}).get(name) == value for name, value in latest_cookies.items()
                        ),
                        "set_cookie": bool(reply.cookies),
                    }
                )
                latest_cookies.update(reply.cookies.items())
            return _AiohttpReply(reply)

        client.post = post
        reply = agent_http.post(
            "/run",
            json={
                "responses_create_params": {
                    "input": [{"role": "user", "content": "synthetic task"}],
                    "tools": [
                        {
                            "type": "function",
                            "name": TOOL,
                            "strict": False,
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                },
                "verifier_metadata": {"servers": ["github"], "enabled_tools": [TOOL], "claims": ["synthetic claim"]},
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
            },
        )
        assert len(model_inputs) == 2
        tool_outputs = [item for item in model_inputs[1]["input"] if item.get("type") == "function_call_output"]
        assert any(
            "authentication failed" in item.get("output", "") and "give your final answer" in item.get("output", "")
            for item in tool_outputs
        )
        # This is explicit test teardown, not evidence that the failed SimpleAgent /run closed automatically.
        resource_http.cookies.clear()
        cleanup = resource_http.post(
            "/close_session",
            json=ResourcesCloseSessionRequest(
                resources_session_id="synthetic-test-cleanup", episode_id=EpisodeId(rollout_id="0-0")
            ).model_dump(mode="json"),
            cookies=latest_cookies,
        )
        assert resource_calls, reply.text
        assert [(call["path"].split("/")[-1], call["status"]) for call in resource_calls] == [
            ("seed_session", 200),
            (TOOL, 422),
            ("verify", 422),
        ]
        assert resource_calls[0]["set_cookie"]
        assert all(call["sent_cookie"] for call in resource_calls[1:])
        assert all(call["sent_latest_cookie"] for call in resource_calls[1:])
        assert len(model_inputs) == 2 and transport.await_count == 1
        judge.post.assert_not_awaited()
        assert reply.status_code == 500
        payload = reply.json()
        assert "authentication failed" in reply.text
        assert isinstance(payload, str)
        assert cleanup.status_code == 200 and not resources._rollouts
