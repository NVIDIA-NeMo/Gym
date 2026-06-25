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
import asyncio
from unittest.mock import AsyncMock, MagicMock

import nemo_gym.base_responses_api_agent as agent_mod
from nemo_gym.base_responses_api_agent import (
    RUN_TOKEN_HEADER,
    AgentRunRequest,
    BaseResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
    to_training_item,
)
from nemo_gym.server_utils import ServerClient


def _msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=f"msg-{text}",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
        role="assistant",
        status="completed",
        type="message",
    )


class TestBaseResponsesAPIAgent:
    def test_BaseResponsesAPIAgent(self) -> None:
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="")
        BaseResponsesAPIAgent(config=config)

    def test_SimpleResponsesAPIAgent(self) -> None:
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="")

        class TestSimpleResponsesAPIAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        agent = TestSimpleResponsesAPIAgent(config=config, server_client=MagicMock(spec=ServerClient))
        agent.setup_webserver()


def _agent() -> SimpleResponsesAPIAgent:
    config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="")

    class _A(SimpleResponsesAPIAgent):
        async def responses(self, body=...):
            raise NotImplementedError

        async def run(self, body=...):
            raise NotImplementedError

    return _A(config=config, server_client=MagicMock(spec=ServerClient))


class TestTokenIdAgentHelpers:
    def test_run_token_from_request(self) -> None:
        req = MagicMock()
        req.headers = {RUN_TOKEN_HEADER: "abc"}
        assert SimpleResponsesAPIAgent.run_token_from_request(req) == "abc"
        req.headers = {}
        assert SimpleResponsesAPIAgent.run_token_from_request(req) is None

    def test_harness_base_url_run_scopes_with_fallback(self) -> None:
        agent = _agent()  # no model_server -> uses the fallback endpoint
        with_token = MagicMock()
        with_token.headers = {RUN_TOKEN_HEADER: "tok"}
        no_token = MagicMock()
        no_token.headers = {}
        assert agent.harness_base_url(with_token, fallback="http://x") == "http://x/runs/tok"
        assert agent.harness_base_url(no_token, fallback="http://x") == "http://x"
        assert agent.harness_base_url(no_token) is None  # no model_server, no fallback


class _RunCfg(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class TestSharedRun:
    """The base run() (online design): mints a token, passes it to /v1/responses via header, and uses
    the model-server-CONSTRUCTED trajectory as gym_resp.output before verifying."""

    def test_run_mints_token_header_and_uses_constructed_trajectory(self, monkeypatch) -> None:
        captured = {}

        class _A(SimpleResponsesAPIAgent):
            config: _RunCfg

            async def responses(self, body=...):
                raise NotImplementedError

            async def get_monotonic_trajectory(self, model_server_name, run_token):
                captured["fetch"] = (model_server_name, run_token)
                # the model server returns a trainable trajectory built from its own view.
                return [to_training_item(_msg("hi"), [9], [8], [-0.1])]

        cfg = _RunCfg(
            host="",
            port=0,
            entrypoint="",
            name="agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="rs"),
            model_server=ModelServerRef(type="responses_api_models", name="ms"),
        )
        agent = _A(config=cfg, server_client=MagicMock(spec=ServerClient))

        resp_obj = NeMoGymResponse(
            id="resp",
            created_at=0,
            model="m",
            object="response",
            output=[_msg("hi")],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
            usage=NeMoGymResponseUsage(
                input_tokens=0,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )

        posts = []

        async def fake_post(server_name, url_path, **kw):
            posts.append((url_path, kw))
            resp = MagicMock()
            resp.cookies = {}
            return resp

        agent.server_client.post = AsyncMock(side_effect=fake_post)
        monkeypatch.setattr(agent_mod, "raise_for_status", AsyncMock())
        # /v1/responses returns the trajectory; /verify returns a full BaseVerifyResponse.
        verify_json = {"responses_create_params": {"input": "hi"}, "response": resp_obj.model_dump(), "reward": 1.0}
        monkeypatch.setattr(
            agent_mod, "get_response_json", AsyncMock(side_effect=[resp_obj.model_dump(), verify_json])
        )

        req = MagicMock()
        req.cookies = {}
        body = AgentRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"))
        result = asyncio.run(agent.run(req, body))

        # the /v1/responses call carried a freshly-minted run token in the header
        responses_kw = next(kw for path, kw in posts if path == "/v1/responses")
        token = responses_kw["headers"][RUN_TOKEN_HEADER]
        assert token and len(token) > 0
        # the constructed-trajectory fetch was invoked with that token + the model_server ref
        assert captured["fetch"] == ("ms", token)
        # and the verify call received the constructed (training) trajectory, not the raw harness output
        verify_kw = next(kw for path, kw in posts if path == "/verify")
        assert "prompt_token_ids" in verify_kw["json"]["response"]["output"][0]
        assert result.reward == 1.0
        assert result.turns_used == 1
        assert result.finished_naturally is True

    def test_run_without_model_server_omits_header(self, monkeypatch) -> None:
        # regression: with no model_server, run() must not pass headers=None (which would trip
        # request()'s json path) and must skip the trajectory fetch.
        class _CfgNoMS(BaseResponsesAPIAgentConfig):
            resources_server: ResourcesServerRef

        class _A(SimpleResponsesAPIAgent):
            config: _CfgNoMS

            async def responses(self, body=...):
                raise NotImplementedError

            async def get_monotonic_trajectory(self, *a, **k):
                raise AssertionError("trajectory fetch must not run without a model_server")

        cfg = _CfgNoMS(
            host="",
            port=0,
            entrypoint="",
            name="agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="rs"),
        )
        agent = _A(config=cfg, server_client=MagicMock(spec=ServerClient))

        resp_obj = NeMoGymResponse(
            id="resp",
            created_at=0,
            model="m",
            object="response",
            output=[_msg("hi")],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
            usage=NeMoGymResponseUsage(
                input_tokens=0,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )

        posts = []

        async def fake_post(server_name, url_path, **kw):
            posts.append((url_path, kw))
            resp = MagicMock()
            resp.cookies = {}
            return resp

        agent.server_client.post = AsyncMock(side_effect=fake_post)
        monkeypatch.setattr(agent_mod, "raise_for_status", AsyncMock())
        verify_json = {"responses_create_params": {"input": "hi"}, "response": resp_obj.model_dump(), "reward": 0.5}
        monkeypatch.setattr(
            agent_mod, "get_response_json", AsyncMock(side_effect=[resp_obj.model_dump(), verify_json])
        )

        req = MagicMock()
        req.cookies = {}
        body = AgentRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"))
        result = asyncio.run(agent.run(req, body))

        responses_kw = next(kw for path, kw in posts if path == "/v1/responses")
        assert "headers" not in responses_kw  # i.e. not headers=None
        assert result.reward == 0.5
