# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.atropos_agent.app import (
    AtroposAgent,
    AtroposAgentConfig,
)
from responses_api_agents.atropos_agent.gym_server_bridge import (
    GymServerManager,
    GymTokenInfo,
    GymVLLMServer,
)


def _make_config(**overrides):
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    return AtroposAgentConfig(**{**defaults, **overrides})


def _mock_vllm_response(content="ok", prompt_ids=None, gen_ids=None, gen_lps=None):
    """Build a mock vLLM chat completion response dict."""
    msg = {"role": "assistant", "content": content}
    if prompt_ids is not None:
        msg["prompt_token_ids"] = prompt_ids
    if gen_ids is not None:
        msg["generation_token_ids"] = gen_ids
    if gen_lps is not None:
        msg["generation_log_probs"] = gen_lps
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _mock_session(response_dict, tokenize_tokens=None):
    """Mock aiohttp session that returns response_dict for /chat/completions
    and tokenize_tokens for /tokenize."""
    if tokenize_tokens is None:
        tokenize_tokens = [1, 2, 3, 4, 5]

    tokenize_response = {"tokens": tokenize_tokens, "count": len(tokenize_tokens)}

    def _make_resp(data):
        r = AsyncMock()
        r.status = 200
        r.json = AsyncMock(return_value=data)
        r.__aenter__ = AsyncMock(return_value=r)
        r.__aexit__ = AsyncMock(return_value=False)
        return r

    def post_side_effect(url, **kwargs):
        if "/tokenize" in url:
            return _make_resp(tokenize_response)
        return _make_resp(response_dict)

    mock_session = MagicMock()
    mock_session.post = MagicMock(side_effect=post_side_effect)
    return mock_session


class TestAtroposAgent:
    def test_construction(self):
        AtroposAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

    def test_concurrency_semaphore(self):
        agent = AtroposAgent(config=_make_config(max_concurrent=16), server_client=MagicMock(spec=ServerClient))
        assert agent._sem is not None
        assert agent._sem._value == 16

    def test_no_env_no_reward_fn(self):
        agent = AtroposAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        assert agent._reward_fn is None
        assert agent._env is None

    def test_build_output_with_token_info(self):
        agent = AtroposAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        ti = GymTokenInfo(prompt_token_ids=[1, 2], generation_token_ids=[3, 4], generation_log_probs=[-0.5, -0.3])
        msgs = [{"role": "user", "content": "q"}]
        output = agent._build_output("answer", ti, msgs)
        assert len(output) == 2
        assert output[1]["prompt_token_ids"] == [1, 2]
        assert output[1]["generation_token_ids"] == [3, 4]

    def test_build_output_without_token_info(self):
        agent = AtroposAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        output = agent._build_output("answer", None, [{"role": "user", "content": "q"}])
        assert len(output) == 2
        assert "prompt_token_ids" not in output[1]


class TestGymVLLMServer:
    def test_initial_state(self):
        server = GymVLLMServer("http://localhost:8000/v1", "test-model")
        assert server.get_token_infos() == []
        assert server.server_healthy is True

    @pytest.mark.asyncio
    async def test_chat_completion_gym_mode(self):
        """Token IDs injected by Gym's vllm_model wrapper."""
        server = GymVLLMServer("http://localhost:8000/v1", "test-model")
        resp_dict = _mock_vllm_response("72", [1, 2, 3], [10, 11], [-0.5, -0.3])
        session = _mock_session(resp_dict)

        with patch(
            "responses_api_agents.atropos_agent.gym_server_bridge.get_global_aiohttp_client", return_value=session
        ):
            result = await server.chat_completion(messages=[{"role": "user", "content": "hi"}])

        assert result.choices[0].message.content == "72"
        ti = server.get_token_infos()
        assert len(ti) == 1
        assert ti[0].prompt_token_ids == [1, 2, 3]
        assert ti[0].generation_token_ids == [10, 11]
        assert ti[0].generation_log_probs == [-0.5, -0.3]

    @pytest.mark.asyncio
    async def test_tools_passthrough(self):
        server = GymVLLMServer("http://localhost:8000/v1", "test-model")
        resp_dict = _mock_vllm_response("ok", [1], [2], [-0.1])
        session = _mock_session(resp_dict)
        tools = [{"type": "function", "function": {"name": "get_weather"}}]

        with patch(
            "responses_api_agents.atropos_agent.gym_server_bridge.get_global_aiohttp_client", return_value=session
        ):
            await server.chat_completion(messages=[{"role": "user", "content": "?"}], tools=tools, tool_choice="auto")

        body = session.post.call_args_list[0].kwargs.get("json", session.post.call_args_list[0][1].get("json", {}))
        assert body.get("tools") == tools
        assert body.get("tool_choice") == "auto"

    @pytest.mark.asyncio
    async def test_clear_token_infos(self):
        server = GymVLLMServer("http://localhost:8000/v1", "test-model")
        resp_dict = _mock_vllm_response("ok", [1], [2], [-0.1])
        session = _mock_session(resp_dict)

        with patch(
            "responses_api_agents.atropos_agent.gym_server_bridge.get_global_aiohttp_client", return_value=session
        ):
            await server.chat_completion(messages=[{"role": "user", "content": "hi"}])

        assert len(server.get_token_infos()) == 1
        server.clear_token_infos()
        assert len(server.get_token_infos()) == 0

    @pytest.mark.asyncio
    async def test_accumulates_across_calls(self):
        """Multi-turn: token infos accumulate across chat_completion calls."""
        server = GymVLLMServer("http://localhost:8000/v1", "test-model")
        resp_dict = _mock_vllm_response("ok", [1, 2], [3], [-0.5])
        session = _mock_session(resp_dict)

        with patch(
            "responses_api_agents.atropos_agent.gym_server_bridge.get_global_aiohttp_client", return_value=session
        ):
            await server.chat_completion(messages=[{"role": "user", "content": "turn 1"}])
            await server.chat_completion(messages=[{"role": "user", "content": "turn 2"}])

        assert len(server.get_token_infos()) == 2


class TestGymServerManager:
    @pytest.mark.asyncio
    async def test_dedicated_server(self):
        manager = GymServerManager("http://localhost:8000/v1", "test-model")
        async with manager.dedicated_server() as server:
            assert isinstance(server, GymVLLMServer)
            assert server.server_healthy

    @pytest.mark.asyncio
    async def test_managed_server(self):
        manager = GymServerManager("http://localhost:8000/v1", "test-model")
        async with manager.managed_server() as server:
            assert isinstance(server, GymVLLMServer)

    def test_servers_list(self):
        manager = GymServerManager("http://localhost:8000/v1", "test-model")
        assert len(manager.servers) == 1
        assert isinstance(manager.servers[0], GymVLLMServer)
