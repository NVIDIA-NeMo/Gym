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
"""Tests for GenRM Compare Resources Server."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch, approx

import resources_servers.genrm_compare.app
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from resources_servers.genrm_compare.app import (
    GenRMCompareConfig,
    GenRMCompareRequest,
    GenRMCompareResourcesServer,
    GenRMCompareResponse,
    GenRMCompareVerifyRequest,
    _input_to_conversation_history,
)
from resources_servers.genrm_compare.utils import get_prompt_key_from_input


class TestGenRMCompareConfig:
    """Test GenRM compare configuration."""

    def test_config_defaults(self):
        """Test configuration with default values."""
        config = GenRMCompareConfig(
            # Required fields from BaseServerConfig
            host="localhost",
            port=8000,
            # Required fields from BaseRunServerConfig
            entrypoint="app.py",
            # Required fields from BaseResourcesServerConfig
            domain="rlhf",
            # GenRMCompareConfig fields
            name="genrm_compare",
            genrm_model_server=ModelServerRef(type="responses_api_models", name="genrm_model"),
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=1024),
        )

        # Check defaults
        assert config.comparison_strategy == "circular"
        assert config.num_judges_per_comparison == 1
        assert config.use_principle is False
        assert config.aggregator_method == "simple_tiebreaker"
        assert config.default_score == 3.0
        assert config.default_ranking == 3.5


class TestGenRMCompareRequest:
    """Test request/response models."""

    def test_request_creation(self):
        """Test creating a compare request."""
        request = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "What is 2+2?"}],
            response_objs=[
                {"output": [{"type": "message", "content": [{"type": "output_text", "text": "4"}]}]},
                {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Four"}]}]},
            ],
            principle="Be concise",
        )

        assert len(request.conversation_history) == 1
        assert len(request.response_objs) == 2
        assert request.principle == "Be concise"

    def test_response_creation(self):
        """Test creating a compare response."""
        response = GenRMCompareResponse(
            rewards=[3.5, 4.0],
            comparison_results=[{"response_i": 0, "response_j": 1, "score_1": 3.0, "score_2": 4.0, "ranking": 4.0}],
            metrics={"mean_individual_score": 3.5},
        )

        assert len(response.rewards) == 2
        assert response.rewards[0] == approx(3.5)
        assert response.rewards[1] == approx(4.0)


class TestGenRMCompareResourcesServer:
    """Test GenRM Compare Resources Server methods."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return GenRMCompareConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            domain="rlhf",
            name="genrm_compare",
            genrm_model_server=ModelServerRef(type="responses_api_models", name="genrm_model"),
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=1024),
            comparison_strategy="circular",
            num_judges_per_comparison=1,
            debug_logging=False,
        )

    def test_single_response_returns_default(self, config):
        """Single response should return default score."""
        # model_construct bypasses Pydantic validation; server_client is unused for single-response path
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())

        # Create request with single response
        request = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "Hello"}], response_objs=[{"output": []}]
        )

        response = asyncio.run(server.compare(request))

        assert len(response.rewards) == 1
        assert response.rewards[0] == config.default_score
        assert response.comparison_results is None
        assert response.metrics is None

    def test_verify_cohort_key_prefers_task_index_then_prompt_id(self, config):
        """Cohort key should use explicit task/prompt identifiers to avoid avoidable collisions."""
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        input_messages = [NeMoGymEasyInputMessage(role="user", content="hello", type="message")]
        prompt_hash = get_prompt_key_from_input(input_messages, "Be concise")

        task_request = GenRMCompareVerifyRequest.model_validate(
            {
                "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input=input_messages),
                "response": NeMoGymResponse(
                    id="resp_task",
                    created_at=0.0,
                    model="dummy_model",
                    tools=[],
                    parallel_tool_calls=True,
                    tool_choice="auto",
                    output=[],
                    object="response",
                ),
                "principle": "Be concise",
                TASK_INDEX_KEY_NAME: 7,
                ROLLOUT_INDEX_KEY_NAME: 2,
            }
        )
        assert task_request.task_index == 7
        assert task_request.rollout_index == 2
        assert server._get_verify_cohort_key(task_request, input_messages, task_request.principle) == (
            f"task_idx::7::{prompt_hash}"
        )

        prompt_request = GenRMCompareVerifyRequest.model_validate(
            {
                "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input=input_messages),
                "response": NeMoGymResponse(
                    id="resp_prompt",
                    created_at=0.0,
                    model="dummy_model",
                    tools=[],
                    parallel_tool_calls=True,
                    tool_choice="auto",
                    output=[],
                    object="response",
                ),
                "principle": "Be concise",
                "prompt_id": "prompt-123",
            }
        )
        assert server._get_verify_cohort_key(prompt_request, input_messages, prompt_request.principle) == (
            f"prompt_id::prompt-123::{prompt_hash}"
        )

    async def test_run_jit_compare_using_most_recent_response_obj(self, monkeypatch: MonkeyPatch) -> None:
        config = GenRMCompareConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            domain="rlhf",
            name="genrm_compare",
            genrm_model_server=ModelServerRef(type="responses_api_models", name="genrm_model"),
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=1024),
            comparison_strategy="circular",
            num_judges_per_comparison=1,
            num_rollouts_per_prompt=16,
            debug_logging=False,
        )
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())

        request = GenRMCompareVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    NeMoGymEasyInputMessage(
                        role="user",
                        content=[{"type": "input_text", "text": "hello"}],
                        type="message",
                    )
                ],
            ),
            response=NeMoGymResponse(
                id="resp_123",
                created_at=0.0,
                model="dummy_model",
                tools=[],
                parallel_tool_calls=True,
                tool_choice="auto",
                output=[
                    NeMoGymResponseReasoningItem(
                        id="rs_123",
                        type="reasoning",
                        summary=[
                            NeMoGymSummary(
                                text="I have identified the city as San Francisco based on user input.",
                                type="summary_text",
                            )
                        ],
                        status="completed",
                    ),
                    NeMoGymResponseOutputMessage(
                        id="msg_123",
                        role="assistant",
                        status="completed",
                        type="message",
                        content=[
                            NeMoGymResponseOutputText(
                                text="hi :) how are you?",
                                type="output_text",
                                annotations=[],
                            )
                        ],
                    ),
                ],
                object="response",
            ),
        )

        # Patch `aggregate_scores`
        aggregate_scores_mock = MagicMock(side_effect=resources_servers.genrm_compare.app.aggregate_scores)
        monkeypatch.setattr(resources_servers.genrm_compare.app, "aggregate_scores", aggregate_scores_mock)

        # Patch `_run_single_comparison`
        async def run_single_comparison_mock(*args, **kwargs):
            i, j = kwargs["pair_idx"]
            # Random deterministic return
            return (5 * (i + 1 / 16), 5 * (j + 1 / 16), 2 if i % 2 else 5)

        monkeypatch.setattr(server, "_run_single_comparison", run_single_comparison_mock)

        golden_result = await server._run_compare(
            conversation_history=_input_to_conversation_history(request.responses_create_params.input),
            response_objs=[request.response.model_dump() for _ in range(16)],
        )
        golden_rewards = golden_result[0]

        tasks = []
        for _ in range(16):
            tasks.append(server.verify(request))

        results = await asyncio.gather(*tasks)

        expected_metadata = (
            (
                0,
                1,
                0,
            ),
            (
                1,
                2,
                0,
            ),
            (
                2,
                3,
                0,
            ),
            (
                3,
                4,
                0,
            ),
            (
                4,
                5,
                0,
            ),
            (
                5,
                6,
                0,
            ),
            (
                6,
                7,
                0,
            ),
            (
                7,
                8,
                0,
            ),
            (
                8,
                9,
                0,
            ),
            (
                9,
                10,
                0,
            ),
            (
                10,
                11,
                0,
            ),
            (
                11,
                12,
                0,
            ),
            (
                12,
                13,
                0,
            ),
            (
                13,
                14,
                0,
            ),
            (
                14,
                15,
                0,
            ),
            (
                15,
                0,
                0,
            ),
        )
        # Call 1 since the second call is our tested call
        actual_metadata = aggregate_scores_mock.call_args_list[1].kwargs["comparison_metadata"]
        assert expected_metadata == actual_metadata

        expected_rewards = golden_rewards
        actual_rewards = [r.reward for r in results]
        assert expected_rewards == actual_rewards


class TestRunSingleComparison:
    """Tests for GenRMCompareResourcesServer._run_single_comparison."""

    def _make_response_obj(self, text):
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}

    def _make_server(self, use_principle=False):
        config = GenRMCompareConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            domain="rlhf",
            name="genrm_compare",
            genrm_model_server=ModelServerRef(type="responses_api_models", name="genrm_model"),
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=1024),
            use_principle=use_principle,
        )
        mock_server_client = MagicMock()
        # Return a well-formed GenRM score response
        mock_http_response = AsyncMock()
        mock_http_response.json = AsyncMock(
            return_value={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"score_1": 4, "score_2": 2, "ranking": 2}'}],
                    }
                ]
            }
        )
        mock_server_client.post = AsyncMock(return_value=mock_http_response)
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=mock_server_client)
        return server, mock_server_client

    def _get_sent_body(self, mock_server_client):
        call_kwargs = mock_server_client.post.call_args.kwargs
        return call_kwargs["json"]

    def test_responses_passed_via_metadata_not_input(self):
        """response_1 and response_2 are sent in metadata, not appended to input."""
        server, mock_client = self._make_server(use_principle=False)
        conversation = [{"role": "user", "content": "What is 2+2?"}]

        asyncio.run(
            server._run_single_comparison(
                conversation,
                self._make_response_obj("4"),
                self._make_response_obj("Four"),
            )
        )

        body = self._get_sent_body(mock_client)
        metadata = body.metadata
        assert metadata["response_1"] == "4"
        assert metadata["response_2"] == "Four"

        # input should contain only the conversation history
        input_roles = [m.role for m in body.input]
        assert input_roles == ["user"]
        assert "response_1" not in input_roles
        assert "response_2" not in input_roles

    def test_principle_passed_via_metadata_when_enabled(self):
        """principle is sent in metadata when use_principle=True."""
        server, mock_client = self._make_server(use_principle=True)
        conversation = [{"role": "user", "content": "Explain gravity."}]

        asyncio.run(
            server._run_single_comparison(
                conversation,
                self._make_response_obj("Gravity pulls objects."),
                self._make_response_obj("Gravity is a force."),
                principle="Be concise.",
            )
        )

        body = self._get_sent_body(mock_client)
        assert body.metadata["principle"] == "Be concise."

    def test_principle_absent_from_metadata_when_disabled(self):
        """principle key is absent from metadata when use_principle=False."""
        server, mock_client = self._make_server(use_principle=False)
        conversation = [{"role": "user", "content": "Hello"}]

        asyncio.run(
            server._run_single_comparison(
                conversation,
                self._make_response_obj("Hi"),
                self._make_response_obj("Hello there"),
                principle="Be concise.",  # ignored when use_principle=False
            )
        )

        body = self._get_sent_body(mock_client)
        assert "principle" not in body.metadata
