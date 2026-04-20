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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.base_gymnasium import EnvResetRequest, EnvStepRequest, GymnasiumServer, extract_text


def _make_response(*parts: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"msg_{i}",
                content=[NeMoGymResponseOutputText(annotations=[], text=p, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
            for i, p in enumerate(parts)
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


class _FakeRequest:
    def __init__(self, session_id="sid-1"):
        self.session = {SESSION_ID_KEY: session_id}


class _TerminatingEnv(GymnasiumServer):
    async def step(self, action, metadata, session_id=None):
        return None, 1.0, True, False, {}


class _OngoingEnv(GymnasiumServer):
    async def step(self, action, metadata, session_id=None):
        return "keep going", 0.0, False, False, {}


class _TruncatingEnv(GymnasiumServer):
    async def step(self, action, metadata, session_id=None):
        return None, 0.0, False, True, {}


_close_log: list = []


class _CustomCloseEnv(GymnasiumServer):
    async def step(self, action, metadata, session_id=None):
        return None, 0.0, True, False, {}

    async def close_session(self, session_id):
        _close_log.append(session_id)
        await super().close_session(session_id)


def _make_env(cls):
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
    return cls(config=config, server_client=MagicMock(spec=ServerClient))


class TestGymnasiumServer:
    def test_routes_registered(self):
        env = _make_env(_TerminatingEnv)
        routes = {r.path for r in env.setup_webserver().routes}
        assert {"/reset", "/step", "/aggregate_metrics"}.issubset(routes)

    def test_verify_raises(self):
        env = _make_env(_TerminatingEnv)
        with pytest.raises(NotImplementedError):
            import asyncio

            asyncio.run(env.verify(SimpleNamespace()))

    @pytest.mark.asyncio
    async def test_reset_default_returns_empty(self):
        env = _make_env(_TerminatingEnv)
        env.session_state["sid-1"] = {"x": 1}
        body = EnvResetRequest(responses_create_params={"input": []})
        resp = await env._reset_endpoint(body, _FakeRequest())
        assert resp.observation is None
        assert resp.info == {}

    @pytest.mark.asyncio
    async def test_step_pops_on_terminated(self):
        env = _make_env(_TerminatingEnv)
        env.session_state["sid-1"] = {"x": 1}
        body = EnvStepRequest(responses_create_params={"input": []}, response=_make_response("x"))
        resp = await env._step_endpoint(body, _FakeRequest("sid-1"))
        assert resp.terminated is True
        assert "sid-1" not in env.session_state

    @pytest.mark.asyncio
    async def test_step_pops_on_truncated(self):
        env = _make_env(_TruncatingEnv)
        env.session_state["sid-1"] = {"x": 1}
        body = EnvStepRequest(responses_create_params={"input": []}, response=_make_response("x"))
        resp = await env._step_endpoint(body, _FakeRequest("sid-1"))
        assert resp.truncated is True
        assert "sid-1" not in env.session_state

    @pytest.mark.asyncio
    async def test_step_keeps_state_when_ongoing(self):
        env = _make_env(_OngoingEnv)
        env.session_state["sid-1"] = {"x": 1}
        body = EnvStepRequest(responses_create_params={"input": []}, response=_make_response("x"))
        resp = await env._step_endpoint(body, _FakeRequest("sid-1"))
        assert resp.terminated is False
        assert resp.truncated is False
        assert "sid-1" in env.session_state

    @pytest.mark.asyncio
    async def test_close_session_override_invoked(self):
        _close_log.clear()
        env = _make_env(_CustomCloseEnv)
        env.session_state["sid-1"] = {"x": 1}
        body = EnvStepRequest(responses_create_params={"input": []}, response=_make_response("x"))
        await env._step_endpoint(body, _FakeRequest("sid-1"))
        assert _close_log == ["sid-1"]
        assert "sid-1" not in env.session_state


class TestExtractText:
    def test_concats_output_text(self):
        r = _make_response("hello ", "world")
        assert extract_text(r) == "hello world"

    def test_empty_output(self):
        r = NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        assert extract_text(r) == ""
