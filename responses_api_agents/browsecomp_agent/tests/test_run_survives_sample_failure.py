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
"""One sample's failure must not abort the whole rollout collection.

`run()` used to log `[browsecomp][abort]` and re-raise. That makes /run return 500, which
the collector treats as fatal (`raise_for_status` in nemo_gym/rollout_collection.py), so a
single bad sample kills every remaining one. It cost two full 8-node allocations:

  * job 5748664 — died at  34/400 (malformed tool-call JSON, fixed separately in bdfba2f4)
  * job 5751899 — died at 167/400 (a 500 out of the agent's own /v1/responses)

So the sample is now scored 0 and the error is carried on the response as `agent_error`,
keeping the failure COUNTABLE in the results rather than silently indistinguishable from a
wrong answer. That distinction matters: if these had been silent zeros, the malformed-JSON
bug would never have been found.
"""

import json as jsonlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.browsecomp_agent.app import (
    BrowsecompAgent,
    BrowsecompAgentConfig,
    BrowsecompAgentRunRequest,
)


def _make_config(**kwargs) -> BrowsecompAgentConfig:
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="test_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="test_resources"),
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        nudge_steps=False,
    )
    return BrowsecompAgentConfig(**(defaults | kwargs))


def _model_response(outputs: list) -> dict:
    return NeMoGymResponse(
        id="resp_001",
        created_at=0.0,
        model="test_model",
        object="response",
        output=outputs,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        usage=NeMoGymResponseUsage(
            input_tokens=0,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        ),
    ).model_dump()


def _answer_msg(text: str = "Exact Answer: X") -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg_001",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


class _RunServerClient:
    """Drives /run: seed_session -> /v1/responses -> /verify, with injectable failures."""

    def __init__(self, responses_status: int = 200, verify_status: int = 200, reward: float = 1.0):
        self.responses_status = responses_status
        self.verify_status = verify_status
        self.reward = reward

    async def post(self, server_name=None, url_path=None, json=None, cookies=None):
        http = MagicMock()
        http.cookies = {}
        http.status = 200
        http.ok = True
        if url_path == "/seed_session":
            http.read = AsyncMock(return_value=b"{}")
            http.content.read = AsyncMock(return_value=b"{}")
        elif url_path == "/v1/responses":
            http.status = self.responses_status
            http.ok = self.responses_status < 400
            payload = _model_response([_answer_msg()])
            http.read = AsyncMock(return_value=jsonlib.dumps(payload).encode())
            http.content.read = AsyncMock(return_value=jsonlib.dumps(payload).encode())
        elif url_path == "/verify":
            http.status = self.verify_status
            http.ok = self.verify_status < 400
            # BaseVerifyResponse extends BaseVerifyRequest, so the real server echoes the
            # whole request back alongside `reward`.
            payload = dict(json or {}) | {"reward": self.reward}
            blob = jsonlib.dumps(payload, default=str).encode()
            http.read = AsyncMock(return_value=blob)
            http.content.read = AsyncMock(return_value=blob)
        else:
            http.read = AsyncMock(return_value=b"{}")
            http.content.read = AsyncMock(return_value=b"{}")
        # raise_for_status() consults .status via the aiohttp API
        if http.status >= 400:
            import aiohttp

            def _raise():
                raise aiohttp.ClientResponseError(
                    request_info=MagicMock(), history=(), status=http.status, message="Internal Server Error"
                )

            http.raise_for_status = _raise
        else:
            http.raise_for_status = MagicMock()
        return http


def _make_agent(client: _RunServerClient, **config_kwargs) -> BrowsecompAgent:
    server_client = MagicMock(spec=ServerClient)
    server_client.post = client.post
    return BrowsecompAgent(config=_make_config(**config_kwargs), server_client=server_client)


def _run_body() -> BrowsecompAgentRunRequest:
    return BrowsecompAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Q?"}]},
            "question": "Q?",
        }
    )


async def _run(agent: BrowsecompAgent):
    request_mock = MagicMock()
    request_mock.cookies = {}
    return await agent.run(request_mock, _run_body())


class TestSampleFailureIsContained:
    async def test_run_does_not_raise_when_responses_500s(self) -> None:
        """The 5751899 failure: /v1/responses 500s. /run must still return a result."""
        agent = _make_agent(_RunServerClient(responses_status=500))
        result = await _run(agent)
        assert result is not None

    async def test_failed_sample_scores_zero(self) -> None:
        agent = _make_agent(_RunServerClient(responses_status=500))
        result = await _run(agent)
        assert result.reward == 0.0

    async def test_failure_is_countable_not_silent(self) -> None:
        """A failed sample must be distinguishable from a genuine wrong answer."""
        agent = _make_agent(_RunServerClient(responses_status=500))
        result = await _run(agent)
        agent_error = getattr(result, "agent_error", None)
        assert agent_error, "failed sample must carry agent_error so it stays countable"
        assert "ClientResponseError" in agent_error

    async def test_verify_failure_is_also_contained(self) -> None:
        """Not just /v1/responses — a /verify 500 must be contained too."""
        agent = _make_agent(_RunServerClient(verify_status=500))
        result = await _run(agent)
        assert result.reward == 0.0
        assert getattr(result, "agent_error", None)


class TestHappyPathUnaffected:
    async def test_successful_run_returns_real_reward(self) -> None:
        agent = _make_agent(_RunServerClient(reward=1.0))
        result = await _run(agent)
        assert result.reward == 1.0
        assert not getattr(result, "agent_error", None), "a clean run must NOT be tagged as an error"

    async def test_genuine_zero_reward_is_not_tagged_as_error(self) -> None:
        """A real wrong answer scores 0 with NO agent_error — that is the distinction."""
        agent = _make_agent(_RunServerClient(reward=0.0))
        result = await _run(agent)
        assert result.reward == 0.0
        assert not getattr(result, "agent_error", None)


@pytest.mark.parametrize("status", [500, 502, 503])
async def test_contained_for_any_server_error(status: int) -> None:
    agent = _make_agent(_RunServerClient(responses_status=status))
    result = await _run(agent)
    assert result.reward == 0.0
