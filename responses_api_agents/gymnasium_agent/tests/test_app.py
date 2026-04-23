# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.gymnasium_agent.app import GymnasiumAgent, GymnasiumAgentConfig, GymnasiumAgentRunRequest


def _make_agent(max_steps=10):
    config = GymnasiumAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="test_gymnasium_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="my_env"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_steps=max_steps,
    )
    return GymnasiumAgent(config=config, server_client=MagicMock(spec=ServerClient))


def _model_response(text: str, input_toks=1, output_toks=1) -> dict:
    return {
        "id": "r",
        "created_at": 0.0,
        "model": "m",
        "object": "response",
        "output": [
            {
                "id": "msg",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": input_toks,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_toks,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_toks + output_toks,
        },
    }


class _FakeHttpResp:
    def __init__(self, payload: dict):
        self._payload = payload
        self.cookies = {}
        self.status = 200
        self.ok = True

    async def json(self):
        return self._payload

    async def read(self):
        return json.dumps(self._payload).encode()

    @property
    def content(self):
        class _Body:
            async def read(inner):
                return json.dumps(self._payload).encode()

        return _Body()

    def raise_for_status(self):
        return None


def _wire_mock_client(agent, responses_per_url):
    """Wire agent.server_client.post to return payloads keyed by url_path."""
    call_log = []

    async def _post(server_name, url_path, json=None, cookies=None, **kw):
        call_log.append((server_name, url_path, json))
        payload = responses_per_url[url_path].pop(0)
        return _FakeHttpResp(payload)

    agent.server_client.post = AsyncMock(side_effect=_post)
    return call_log


class TestRoutes:
    def test_routes_registered(self):
        app = _make_agent().setup_webserver()
        routes = {r.path for r in app.routes}
        assert {"/run", "/v1/responses", "/aggregate_metrics"}.issubset(routes)


class TestConfig:
    def test_max_steps_validator_rejects_zero(self):
        with pytest.raises(Exception):
            GymnasiumAgentConfig(
                host="",
                port=0,
                entrypoint="",
                name="x",
                resources_server=ResourcesServerRef(type="resources_servers", name="e"),
                model_server=ModelServerRef(type="responses_api_models", name="m"),
                max_steps=0,
            )

    def test_default_max_steps(self):
        assert _make_agent().config.max_steps == 10


class TestRun:
    @pytest.mark.asyncio
    async def test_terminates_on_first_step(self):
        agent = _make_agent()
        call_log = _wire_mock_client(
            agent,
            {
                "/reset": [{"observation": "go", "info": {}}],
                "/v1/responses": [_model_response("move A")],
                "/step": [{"observation": None, "reward": 1.0, "terminated": True, "truncated": False, "info": {}}],
            },
        )
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "play"}]})
        result = await agent.run(req, body)
        assert result.terminated is True
        assert result.reward == 1.0
        # reset + exactly 1 model call + 1 step
        urls = [u for (_s, u, _) in call_log]
        assert urls.count("/reset") == 1
        assert urls.count("/v1/responses") == 1
        assert urls.count("/step") == 1

    @pytest.mark.asyncio
    async def test_multi_step_preserves_output_items_in_history(self):
        agent = _make_agent(max_steps=3)
        call_log = _wire_mock_client(
            agent,
            {
                "/reset": [{"observation": "start", "info": {}}],
                "/v1/responses": [
                    _model_response("turn-1", output_toks=10),
                    _model_response("turn-2", output_toks=20),
                ],
                "/step": [
                    {"observation": "obs-1", "reward": 0.5, "terminated": False, "truncated": False, "info": {}},
                    {"observation": None, "reward": 0.5, "terminated": True, "truncated": False, "info": {}},
                ],
            },
        )
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "play"}]})
        result = await agent.run(req, body)
        assert result.reward == 1.0
        assert result.terminated is True
        # Inspect turn-2 model call body: its input must contain the full turn-1 output item,
        # not a flattened string, and the obs-1 appended as user message.
        turn2_body = [body for (s, u, body) in call_log if u == "/v1/responses"][1]
        turn2_input = turn2_body.input
        # turn-1 full output item preserved (with structured content list)
        assistant_items = [m for m in turn2_input if getattr(m, "role", None) == "assistant"]
        assert any(
            isinstance(getattr(m, "content", None), list)
            and any(
                getattr(c, "type", None) == "output_text" and getattr(c, "text", "") == "turn-1" for c in m.content
            )
            for m in assistant_items
        ), f"turn-1 output not preserved in structured form: {assistant_items}"
        # obs-1 appended as a user message after turn-1
        assert any(getattr(m, "role", None) == "user" and getattr(m, "content", "") == "obs-1" for m in turn2_input)

    @pytest.mark.asyncio
    async def test_max_steps_sets_truncated(self):
        agent = _make_agent(max_steps=2)
        _wire_mock_client(
            agent,
            {
                "/reset": [{"observation": None, "info": {}}],
                "/v1/responses": [_model_response("a"), _model_response("b")],
                "/step": [
                    {"observation": "obs-1", "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
                    {"observation": "obs-2", "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
                ],
            },
        )
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "x"}]})
        result = await agent.run(req, body)
        assert result.truncated is True
        assert result.terminated is False

    @pytest.mark.asyncio
    async def test_usage_accumulates_across_turns(self):
        agent = _make_agent(max_steps=3)
        _wire_mock_client(
            agent,
            {
                "/reset": [{"observation": None, "info": {}}],
                "/v1/responses": [
                    _model_response("a", input_toks=5, output_toks=7),
                    _model_response("b", input_toks=11, output_toks=13),
                ],
                "/step": [
                    {"observation": "o", "reward": 0.0, "terminated": False, "truncated": False, "info": {}},
                    {"observation": None, "reward": 0.0, "terminated": True, "truncated": False, "info": {}},
                ],
            },
        )
        req = MagicMock()
        req.cookies = {}
        body = GymnasiumAgentRunRequest(responses_create_params={"input": [{"role": "user", "content": "x"}]})
        result = await agent.run(req, body)
        # usage summed across both turns
        assert result.response.usage.input_tokens == 16
        assert result.response.usage.output_tokens == 20
        assert result.response.usage.total_tokens == 36
