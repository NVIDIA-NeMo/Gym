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

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.langchain_deepagents_agent.app import DeepAgentsAgent
from responses_api_agents.langchain_deepagents_agent.reasoning_search_agent import (
    ReasoningSearchDeepAgent,
    ReasoningSearchDeepAgentConfig,
)


def _config(**kwargs) -> ReasoningSearchDeepAgentConfig:
    kwargs.setdefault("resources_server", ResourcesServerRef(type="resources_servers", name="reasoning_gym"))
    kwargs.setdefault("model_server", ModelServerRef(type="responses_api_models", name="policy_model"))
    kwargs.setdefault("tavily_api_key", "test-tavily-key")
    return ReasoningSearchDeepAgentConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **kwargs)


def _make_agent(**kwargs) -> ReasoningSearchDeepAgent:
    return ReasoningSearchDeepAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))


# --- build_agent abstractness / concreteness --------------------------------------------------------


def test_base_class_build_agent_is_abstract():
    with pytest.raises(NotImplementedError):
        DeepAgentsAgent.build_agent(MagicMock(), MagicMock())


def test_concrete_agent_builds_successfully():
    agent = _make_agent()
    assert agent.agent is not None


# --- cookie propagation ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_responses_propagates_inbound_cookies_to_outgoing_response():
    from fastapi import Response

    agent = _make_agent()
    agent.agent = MagicMock()
    agent.agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="done")]})

    request = MagicMock()
    request.cookies = {"session": "abc123"}
    request.path_params = {}
    response = Response()

    body = MagicMock()
    body.input = "hello"

    await agent.responses(request, response, body)
    assert response.raw_headers  # a Set-Cookie header was added
    assert any(b"session=abc123" in value for _, value in response.raw_headers)
