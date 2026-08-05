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

from nemo_gym.base_resources_server import (
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


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

    def test_build_skipped_verify_response_payload(self) -> None:
        config = BaseResponsesAPIAgentConfig(
            host="",
            port=0,
            entrypoint="",
            name="",
            skip_verification=True,
            skip_verification_reward=0.75,
        )

        class TestSimpleResponsesAPIAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        agent = TestSimpleResponsesAPIAgent(config=config, server_client=MagicMock(spec=ServerClient))
        body = BaseRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]))
        payload = agent.build_skipped_verify_response_payload(body, {"id": "response_id"})

        assert payload == {
            "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input=[]).model_dump(),
            "response": {"id": "response_id"},
            "reward": 0.75,
            "verification_skipped": True,
        }

    async def test_call_verify_or_skip_calls_resources_server(self) -> None:
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="")

        class TestSimpleResponsesAPIAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        server_client = MagicMock(spec=ServerClient)
        verify_response = AsyncMock()
        verify_response.ok = True

        response_payload = {
            "id": "response_id",
            "created_at": 1,
            "model": "dummy_model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        body = BaseRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]))
        verify_response.read.return_value = json.dumps(
            body.model_dump() | {"response": response_payload, "reward": 1.0}
        ).encode()
        server_client.post = AsyncMock(return_value=verify_response)

        agent = TestSimpleResponsesAPIAgent(config=config, server_client=server_client)
        result = await agent.call_verify_or_skip(
            body=body,
            response=response_payload,
            resources_server_name="resources_server",
            verify_request_type=BaseVerifyRequest,
            verify_response_type=BaseVerifyResponse,
            cookies={"session": "cookie"},
        )

        assert result.reward == 1.0
        server_client.post.assert_called_once_with(
            server_name="resources_server",
            url_path="/verify",
            json=BaseVerifyRequest.model_validate(body.model_dump() | {"response": response_payload}).model_dump(),
            cookies={"session": "cookie"},
        )

    async def test_call_verify_or_skip_skips_resources_server(self) -> None:
        config = BaseResponsesAPIAgentConfig(
            host="",
            port=0,
            entrypoint="",
            name="",
            skip_verification=True,
            skip_verification_reward=0.5,
        )

        class TestSimpleResponsesAPIAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock()
        agent = TestSimpleResponsesAPIAgent(config=config, server_client=server_client)
        body = BaseRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]))

        response_payload = {
            "id": "response_id",
            "created_at": 1,
            "model": "dummy_model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        result = await agent.call_verify_or_skip(
            body=body,
            response=response_payload,
            resources_server_name="resources_server",
            verify_request_type=BaseVerifyRequest,
            verify_response_type=BaseVerifyResponse,
        )

        assert result.reward == 0.5
        server_client.post.assert_not_called()

    async def test_aggregate_metrics_skip_verification_warns_and_returns_empty_metrics(self) -> None:
        config = BaseResponsesAPIAgentConfig(
            host="",
            port=0,
            entrypoint="",
            name="",
            skip_verification=True,
        )

        class TestSimpleResponsesAPIAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=...):
                raise NotImplementedError

            async def run(self, body=...):
                raise NotImplementedError

        agent = TestSimpleResponsesAPIAgent(config=config, server_client=MagicMock(spec=ServerClient))
        body = AggregateMetricsRequest(verify_responses=[])

        with pytest.warns(RuntimeWarning, match="skip_verification=True"):
            result = await agent.aggregate_metrics(body)

        assert result.group_level_metrics == []
        assert result.agent_metrics == {}
        assert result.key_metrics == {}
