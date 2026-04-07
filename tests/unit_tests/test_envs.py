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

from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import BaseResourcesServerConfig, BaseVerifyRequest
from nemo_gym.envs import Env, EnvResetRequest, EnvResetResponse, EnvStepRequest, EnvStepResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


def _config():
    return BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")


def _routes(server):
    app = server.setup_webserver()
    return {r.path for r in app.routes}


def _mock_request(session_id="test-session"):
    request = MagicMock()
    request.session = {SESSION_ID_KEY: session_id}
    return request


class ConcreteEnv(Env):
    async def step(self, action, metadata, session_id=None):
        return None, 1.0, True, False, {}


def _make_env():
    return ConcreteEnv(config=_config(), server_client=MagicMock(spec=ServerClient))


class TestEnv:
    def test_routes(self):
        routes = _routes(_make_env())
        assert "/reset" in routes
        assert "/step" in routes
        assert "/aggregate_metrics" in routes
        assert "/verify" not in routes
        assert "/seed_session" not in routes

    @pytest.mark.asyncio
    async def test_default_reset_returns_none_observation(self):
        env = _make_env()
        obs, info = await env.reset({})
        assert obs is None
        assert info == {}

    @pytest.mark.asyncio
    async def test_reset_endpoint(self):
        env = _make_env()
        body = MagicMock(spec=EnvResetRequest)
        body.model_extra = {}
        result = await env._reset_endpoint(body, _mock_request())
        assert isinstance(result, EnvResetResponse)
        assert result.observation is None

    @pytest.mark.asyncio
    async def test_step_endpoint_returns_env_step_response(self):
        env = _make_env()
        body = MagicMock(spec=EnvStepRequest)
        body.model_extra = {}
        body.response = MagicMock()
        result = await env._step_endpoint(body, _mock_request())
        assert isinstance(result, EnvStepResponse)
        assert result.terminated is True
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_step_terminated_fields(self):
        class TerminatingEnv(Env):
            async def step(self, action, metadata, session_id=None):
                return None, 0.5, True, False, {"extra": "data"}

        env = TerminatingEnv(config=_config(), server_client=MagicMock(spec=ServerClient))
        body = MagicMock(spec=EnvStepRequest)
        body.model_extra = {}
        body.response = MagicMock()
        result = await env._step_endpoint(body, _mock_request())
        assert result.reward == 0.5
        assert result.terminated is True
        assert result.truncated is False
        assert result.info == {"extra": "data"}

    @pytest.mark.asyncio
    async def test_step_continues_with_observation(self):
        class ContinuingEnv(Env):
            async def step(self, action, metadata, session_id=None):
                return "follow up", 0.0, False, False, {}

        env = ContinuingEnv(config=_config(), server_client=MagicMock(spec=ServerClient))
        body = MagicMock(spec=EnvStepRequest)
        body.model_extra = {}
        body.response = MagicMock()
        result = await env._step_endpoint(body, _mock_request())
        assert result.observation == "follow up"
        assert result.terminated is False

    @pytest.mark.asyncio
    async def test_verify_raises(self):
        env = _make_env()
        with pytest.raises(NotImplementedError):
            await env.verify(MagicMock(spec=BaseVerifyRequest))

    @pytest.mark.asyncio
    async def test_metadata_passed_to_step(self):
        received = {}

        class MetaEnv(Env):
            async def step(self, action, metadata, session_id=None):
                received.update(metadata)
                return None, 0.0, True, False, {}

        env = MetaEnv(config=_config(), server_client=MagicMock(spec=ServerClient))
        body = MagicMock(spec=EnvStepRequest)
        body.model_extra = {"expected_answer": "42"}
        body.response = MagicMock()
        await env._step_endpoint(body, _mock_request())
        assert received.get("expected_answer") == "42"

    @pytest.mark.asyncio
    async def test_session_id_passed_to_step(self):
        received = {}

        class SessionEnv(Env):
            async def step(self, action, metadata, session_id=None):
                received["session_id"] = session_id
                return None, 0.0, True, False, {}

        env = SessionEnv(config=_config(), server_client=MagicMock(spec=ServerClient))
        body = MagicMock(spec=EnvStepRequest)
        body.model_extra = {}
        body.response = MagicMock()
        await env._step_endpoint(body, _mock_request("my-session"))
        assert received.get("session_id") == "my-session"

    @pytest.mark.asyncio
    async def test_session_id_passed_to_reset(self):
        received = {}

        class SessionEnv(Env):
            async def reset(self, metadata, session_id=None):
                received["session_id"] = session_id
                return None, {}

            async def step(self, action, metadata, session_id=None):
                return None, 0.0, True, False, {}

        env = SessionEnv(config=_config(), server_client=MagicMock(spec=ServerClient))
        body = MagicMock(spec=EnvResetRequest)
        body.model_extra = {}
        await env._reset_endpoint(body, _mock_request("my-session"))
        assert received.get("session_id") == "my-session"
