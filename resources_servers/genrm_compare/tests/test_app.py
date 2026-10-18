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
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pytest import MonkeyPatch, approx

import resources_servers.genrm_compare.app
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import (
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.judge import JudgeError
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
    GROUP_ATTEMPT_KEY_NAME,
    GenRMCompareConfig,
    GenRMCompareRequest,
    GenRMCompareResourcesServer,
    GenRMCompareResponse,
    GenRMCompareVerifyRequest,
    _input_to_conversation_history,
    _output_budget_exhausted,
)
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import collected_comparisons
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
        assert config.comparison_mode == "rollout_cohort"
        assert config.comparison_strategy == "circular"
        assert config.num_judges_per_comparison == 1
        assert config.cohort_collection_timeout_s == 1800.0
        assert config.cohort_result_ttl_s == 3600.0
        assert config.max_terminal_cohorts == 4096
        assert config.use_principle is False
        assert config.aggregator_method == "simple_tiebreaker"
        assert config.score_source == "overall"
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

    @staticmethod
    def _verify_request(
        rollout_index: int | None,
        *,
        task_index: int | None = 1,
        group_id: str | None = None,
        group_attempt: int = 0,
        response_id: str | None = None,
    ) -> GenRMCompareVerifyRequest:
        return GenRMCompareVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="hello", type="message")]
            ),
            response=NeMoGymResponse(
                id=response_id or f"resp_{rollout_index}",
                created_at=0.0,
                model="dummy_model",
                tools=[],
                parallel_tool_calls=True,
                tool_choice="auto",
                output=[],
                object="response",
            ),
            task_index=task_index,
            group_id=group_id,
            group_attempt=group_attempt,
            rollout_index=rollout_index,
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

    def test_verify_cohort_key_prefers_group_id_then_task_index(self, config):
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
            f"task_idx::7::{prompt_hash}::group_attempt::0"
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
            f"prompt_id::prompt-123::{prompt_hash}::group_attempt::0"
        )

        group_request = self._verify_request(2, task_index=None, group_id="legacy-group")
        assert server._get_verify_cohort_key(group_request, input_messages) == (
            "group_id::legacy-group::group_attempt::0"
        )

        explicit_group_request = self._verify_request(2, task_index=7, group_id="stable-group")
        assert server._get_verify_cohort_key(explicit_group_request, input_messages) == (
            "group_id::stable-group::group_attempt::0"
        )

    def test_verify_request_accepts_group_attempt_alias(self):
        request = GenRMCompareVerifyRequest.model_validate(
            self._verify_request(0).model_dump() | {GROUP_ATTEMPT_KEY_NAME: 3}
        )

        assert request.group_attempt == 3

    def test_verify_request_defaults_missing_group_attempt_to_zero_with_warning(self, caplog):
        payload = self._verify_request(0, task_index=None, group_id="legacy-group").model_dump(by_alias=True)
        payload.pop(GROUP_ATTEMPT_KEY_NAME)

        resources_servers.genrm_compare.app._warn_legacy_attempt.cache_clear()
        request = GenRMCompareVerifyRequest.model_validate(payload)
        assert "group attempt zero" in caplog.text

        assert request.group_attempt == 0

    def test_verify_request_without_group_id_needs_no_group_attempt(self):
        payload = self._verify_request(0, task_index=7).model_dump(by_alias=True)
        payload.pop(GROUP_ATTEMPT_KEY_NAME)

        request = GenRMCompareVerifyRequest.model_validate(payload)

        assert request.group_id is None
        assert request.group_attempt == 0

    async def test_verify_full_cohort_matches_compare(self, monkeypatch: MonkeyPatch) -> None:
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
            # Make each pair deterministic and distinguishable.
            scores = (5 * (i + 1 / 16), 5 * (j + 1 / 16), 2 if i % 2 else 5)
            # Selected scores, overall scores, unavailable token metrics, and failure flags.
            return (*scores, *scores, -1.0, -1.0, -1.0, 0.0, 0.0)

        monkeypatch.setattr(server, "_run_single_comparison", run_single_comparison_mock)

        golden_result = await server._run_compare(
            conversation_history=_input_to_conversation_history(request.responses_create_params.input),
            response_objs=[request.response.model_dump() for _ in range(16)],
        )
        golden_rewards = golden_result[0]
        aggregate_scores_mock.reset_mock()

        tasks = []
        for rollout_index in range(16):
            tasks.append(
                server.verify(
                    request.model_copy(
                        update={
                            "task_index": 7,
                            "rollout_index": rollout_index,
                        }
                    )
                )
            )

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
        actual_metadata = aggregate_scores_mock.call_args_list[0].kwargs["comparison_metadata"]
        assert expected_metadata == tuple(actual_metadata)

        expected_rewards = golden_rewards
        actual_rewards = [r.reward for r in results]
        assert expected_rewards == actual_rewards
        assert all(r.reasoning_text.startswith("I have identified") for r in results)
        assert all(r.answer_text == "hi :) how are you?" for r in results)

    async def test_verify_maps_rewards_by_rollout_index_not_arrival_order(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(update={"num_rollouts_per_prompt": 3})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())

        def request(rollout_index: int) -> GenRMCompareVerifyRequest:
            return GenRMCompareVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(role="user", content="hello", type="message")]
                ),
                response=NeMoGymResponse(
                    id=f"resp_{rollout_index}",
                    created_at=0.0,
                    model="dummy_model",
                    tools=[],
                    parallel_tool_calls=True,
                    tool_choice="auto",
                    output=[],
                    object="response",
                ),
                task_index=11,
                rollout_index=rollout_index,
            )

        run_compare = AsyncMock(return_value=collected_comparisons([10.0, 20.0, 30.0]))
        monkeypatch.setattr(server, "_collect_comparisons", run_compare)

        results = await asyncio.gather(*(server.verify(request(index)) for index in (2, 0, 1)))

        assert [result.reward for result in results] == [30.0, 10.0, 20.0]
        run_compare.assert_awaited_once()
        response_ids = [response_obj["id"] for response_obj in run_compare.await_args.kwargs["response_objs"]]
        assert response_ids == ["resp_0", "resp_1", "resp_2"]

    async def test_identical_duplicate_attaches_to_existing_rollout_slot(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())

        def request(rollout_index: int) -> GenRMCompareVerifyRequest:
            return GenRMCompareVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(role="user", content="hello", type="message")]
                ),
                response=NeMoGymResponse(
                    id=f"resp_{rollout_index}",
                    created_at=0.0,
                    model="dummy_model",
                    tools=[],
                    parallel_tool_calls=True,
                    tool_choice="auto",
                    output=[],
                    object="response",
                ),
                task_index=12,
                rollout_index=rollout_index,
            )

        run_compare = AsyncMock(return_value=collected_comparisons([1.0, 2.0]))
        monkeypatch.setattr(server, "_collect_comparisons", run_compare)
        first = request(0)

        results = await asyncio.gather(server.verify(first), server.verify(first), server.verify(request(1)))

        assert [result.reward for result in results] == [1.0, 1.0, 2.0]
        run_compare.assert_awaited_once()
        assert server._verify_cohorts == {}

    async def test_conflicting_duplicate_is_rejected_without_growing_cohort(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())

        def request(rollout_index: int, response_id: str) -> GenRMCompareVerifyRequest:
            return GenRMCompareVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(role="user", content="hello", type="message")]
                ),
                response=NeMoGymResponse(
                    id=response_id,
                    created_at=0.0,
                    model="dummy_model",
                    tools=[],
                    parallel_tool_calls=True,
                    tool_choice="auto",
                    output=[],
                    object="response",
                ),
                task_index=13,
                rollout_index=rollout_index,
            )

        run_compare = AsyncMock(return_value=collected_comparisons([1.0, 2.0]))
        monkeypatch.setattr(server, "_collect_comparisons", run_compare)
        original = asyncio.create_task(server.verify(request(0, "original")))
        await asyncio.sleep(0)

        with pytest.raises(HTTPException) as error:
            await server.verify(request(0, "replacement"))

        assert error.value.status_code == 409
        assert len(next(iter(server._verify_cohorts.values())).members) == 1
        second = asyncio.create_task(server.verify(request(1, "sibling")))
        results = await asyncio.gather(original, second)
        assert [result.reward for result in results] == [1.0, 2.0]
        assert server._verify_cohorts == {}

    async def test_verify_supports_legacy_prompt_only_cohort(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        run_compare = AsyncMock(return_value=collected_comparisons([1.0, 2.0]))
        monkeypatch.setattr(server, "_collect_comparisons", run_compare)

        results = await asyncio.gather(
            server.verify(self._verify_request(0, task_index=None)),
            server.verify(self._verify_request(1, task_index=None)),
        )

        assert [result.reward for result in results] == [1.0, 2.0]
        run_compare.assert_awaited_once()
        assert server._verify_cohorts == {}

    async def test_completed_legacy_cohort_does_not_block_sequential_rerun(
        self,
        config,
        monkeypatch: MonkeyPatch,
    ):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        run_compare = AsyncMock(
            side_effect=[
                collected_comparisons([1.0, 2.0]),
                collected_comparisons([3.0, 4.0]),
            ]
        )
        monkeypatch.setattr(server, "_collect_comparisons", run_compare)

        first = await asyncio.gather(
            server.verify(self._verify_request(0, response_id="first-0")),
            server.verify(self._verify_request(1, response_id="first-1")),
        )
        second = await asyncio.gather(
            server.verify(self._verify_request(0, response_id="second-0")),
            server.verify(self._verify_request(1, response_id="second-1")),
        )

        assert [result.reward for result in first] == [1.0, 2.0]
        assert [result.reward for result in second] == [3.0, 4.0]
        assert run_compare.await_count == 2
        assert server._verify_cohorts == {}

    async def test_verify_rejects_invalid_logical_coordinates(self, config):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())

        with pytest.raises(HTTPException) as missing_rollout_index:
            await server.verify(self._verify_request(None))
        with pytest.raises(HTTPException) as invalid_rollout_index:
            await server.verify(self._verify_request(2))

        assert missing_rollout_index.value.status_code == 422
        assert invalid_rollout_index.value.status_code == 422
        assert server._verify_cohorts == {}

    async def test_late_identical_duplicate_receives_cached_reward(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        run_compare = AsyncMock(return_value=collected_comparisons([1.0, 2.0]))
        monkeypatch.setattr(server, "_collect_comparisons", run_compare)
        first = self._verify_request(0, task_index=None, group_id="legacy-21")

        await asyncio.gather(
            server.verify(first),
            server.verify(self._verify_request(1, task_index=None, group_id="legacy-21")),
        )
        duplicate = await server.verify(first)

        assert duplicate.reward == 1.0
        run_compare.assert_awaited_once()
        cohort = next(iter(server._verify_cohorts.values()))
        assert all(member.body is None and not member.waiters for member in cohort.members.values())

    async def test_new_group_attempt_is_isolated_from_completed_cohort(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        run_compare = AsyncMock(
            side_effect=[
                collected_comparisons([1.0, 2.0]),
                collected_comparisons([3.0, 4.0]),
            ]
        )
        monkeypatch.setattr(server, "_collect_comparisons", run_compare)

        first_attempt = await asyncio.gather(
            server.verify(self._verify_request(0, task_index=None, group_id="completed-group", response_id="old-0")),
            server.verify(self._verify_request(1, task_index=None, group_id="completed-group", response_id="old-1")),
        )
        replacement_attempt = await asyncio.gather(
            server.verify(
                self._verify_request(
                    0,
                    task_index=None,
                    group_id="completed-group",
                    group_attempt=1,
                    response_id="replacement-0",
                )
            ),
            server.verify(
                self._verify_request(
                    1,
                    task_index=None,
                    group_id="completed-group",
                    group_attempt=1,
                    response_id="replacement-1",
                )
            ),
        )

        assert [result.reward for result in first_attempt] == [1.0, 2.0]
        assert [result.reward for result in replacement_attempt] == [3.0, 4.0]
        assert run_compare.await_count == 2
        assert len(server._verify_cohorts) == 2

    async def test_partial_old_attempt_does_not_mix_with_completed_replacement(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        monkeypatch.setattr(
            server,
            "_collect_comparisons",
            AsyncMock(return_value=collected_comparisons([3.0, 4.0])),
        )

        old_waiter = asyncio.create_task(
            server.verify(
                self._verify_request(
                    0,
                    task_index=None,
                    group_id="replacement-group",
                    group_attempt=0,
                    response_id="old-0",
                )
            )
        )
        await asyncio.sleep(0)
        replacement = await asyncio.gather(
            server.verify(
                self._verify_request(
                    0,
                    task_index=None,
                    group_id="replacement-group",
                    group_attempt=1,
                    response_id="new-0",
                )
            ),
            server.verify(
                self._verify_request(
                    1,
                    task_index=None,
                    group_id="replacement-group",
                    group_attempt=1,
                    response_id="new-1",
                )
            ),
        )

        old_result = await asyncio.gather(old_waiter, return_exceptions=True)

        assert [result.reward for result in replacement] == [3.0, 4.0]
        assert [result.group_attempt for result in replacement] == [1, 1]
        assert len(server._verify_cohorts) == 2
        assert len(old_result) == 1
        assert isinstance(old_result[0], HTTPException)
        assert old_result[0].status_code == 503
        assert "superseded by attempt 1" in str(old_result[0].detail)
        old_cohort = next(cohort for cohort in server._verify_cohorts.values() if cohort.group_attempt == 0)
        assert old_cohort.phase == "failed"
        assert all(member.body is None and not member.waiters for member in old_cohort.members.values())

    async def test_late_older_group_attempt_is_rejected(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        monkeypatch.setattr(
            server,
            "_collect_comparisons",
            AsyncMock(return_value=collected_comparisons([3.0, 4.0])),
        )

        replacement = await asyncio.gather(
            server.verify(
                self._verify_request(
                    0,
                    task_index=None,
                    group_id="stale-group",
                    group_attempt=1,
                    response_id="new-0",
                )
            ),
            server.verify(
                self._verify_request(
                    1,
                    task_index=None,
                    group_id="stale-group",
                    group_attempt=1,
                    response_id="new-1",
                )
            ),
        )

        with pytest.raises(HTTPException) as error:
            await server.verify(
                self._verify_request(
                    0,
                    task_index=None,
                    group_id="stale-group",
                    group_attempt=0,
                    response_id="old-0",
                )
            )

        assert [result.reward for result in replacement] == [3.0, 4.0]
        assert error.value.status_code == 409
        assert "superseded by attempt 1" in str(error.value.detail)
        assert server._latest_group_attempts["stale-group"].latest_attempt == 1

    async def test_prompt_mismatch_cannot_supersede_active_group_attempt(self, config):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        first_waiter = asyncio.create_task(
            server.verify(
                self._verify_request(
                    0,
                    task_index=None,
                    group_id="stable-group",
                    group_attempt=0,
                )
            )
        )
        await asyncio.sleep(0)
        mismatched = self._verify_request(
            0,
            task_index=None,
            group_id="stable-group",
            group_attempt=1,
        ).model_copy(
            update={
                "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(
                    input=[
                        NeMoGymEasyInputMessage(
                            role="user",
                            content="different prompt",
                            type="message",
                        )
                    ]
                )
            }
        )

        with pytest.raises(HTTPException) as error:
            await server.verify(mismatched)

        assert error.value.status_code == 409
        assert "inconsistent prompt" in str(error.value.detail)
        assert server._latest_group_attempts["stable-group"].latest_attempt == 0
        cohort = next(iter(server._verify_cohorts.values()))
        assert cohort.phase == "collecting"
        first_waiter.cancel()
        await asyncio.gather(first_waiter, return_exceptions=True)

    async def test_prompt_mismatch_in_group_attempt_is_rejected(self, config):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        first = self._verify_request(0, task_index=None, group_id="prompt-mismatch", group_attempt=0)
        first_waiter = asyncio.create_task(server.verify(first))
        await asyncio.sleep(0)
        mismatched = self._verify_request(1, task_index=None, group_id="prompt-mismatch", group_attempt=0).model_copy(
            update={
                "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(role="user", content="different prompt", type="message")]
                )
            }
        )

        with pytest.raises(HTTPException) as error:
            await server.verify(mismatched)

        assert error.value.status_code == 409
        assert "inconsistent prompt" in error.value.detail
        first_waiter.cancel()
        await asyncio.gather(first_waiter, return_exceptions=True)

    async def test_incomplete_cohort_times_out_and_releases_waiters(self, config):
        config = config.model_copy(
            update={
                "num_rollouts_per_prompt": 2,
                "cohort_collection_timeout_s": 0.01,
            }
        )
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())

        waiter = asyncio.create_task(server.verify(self._verify_request(0, task_index=22)))
        await asyncio.sleep(0)
        cohort = next(iter(server._verify_cohorts.values()))
        with pytest.raises(HTTPException, match="did not collect 2 unique rollout indices") as error:
            await asyncio.wait_for(waiter, timeout=1.0)

        assert error.value.status_code == 503
        assert server._verify_cohorts[cohort.key] is cohort
        assert cohort.phase == "failed"
        assert cohort.collection_timeout_task is None
        assert all(member.body is None and not member.waiters for member in cohort.members.values())

    async def test_disconnected_waiter_does_not_retire_logical_cohort(self, config):
        config = config.model_copy(
            update={
                "num_rollouts_per_prompt": 2,
                "cohort_collection_timeout_s": 1.0,
            }
        )
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        waiter = asyncio.create_task(server.verify(self._verify_request(0, task_index=27)))
        await asyncio.sleep(0)
        cohort = next(iter(server._verify_cohorts.values()))

        assert cohort.collection_timeout_task is not None
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)

        assert cohort.phase == "collecting"
        assert len(cohort.members) == 1
        assert all(member.body is not None and not member.waiters for member in cohort.members.values())

    async def test_evaluation_failure_releases_every_waiter(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        monkeypatch.setattr(server, "_collect_comparisons", AsyncMock(side_effect=ValueError("judge failed")))

        results = await asyncio.gather(
            server.verify(self._verify_request(0, task_index=23)),
            server.verify(self._verify_request(1, task_index=23)),
            return_exceptions=True,
        )

        assert len(results) == 2
        assert all(
            isinstance(result, HTTPException) and result.status_code == 503 and "judge failed" in str(result.detail)
            for result in results
        )
        cohort = next(iter(server._verify_cohorts.values()))
        assert cohort.phase == "failed"
        assert all(member.body is None and not member.waiters for member in cohort.members.values())

    async def test_input_materialization_failure_releases_every_waiter(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        run_compare = AsyncMock()
        monkeypatch.setattr(server, "_collect_comparisons", run_compare)
        monkeypatch.setattr(server, "_response_digest", lambda response: response.id)

        def fail_model_dump(*args, **kwargs):
            raise ValueError("response conversion failed")

        monkeypatch.setattr(NeMoGymResponse, "model_dump", fail_model_dump)
        results = await asyncio.gather(
            server.verify(self._verify_request(0, task_index=26)),
            server.verify(self._verify_request(1, task_index=26)),
            return_exceptions=True,
        )

        assert all(
            isinstance(result, HTTPException)
            and result.status_code == 503
            and "response conversion failed" in str(result.detail)
            for result in results
        )
        run_compare.assert_not_awaited()
        cohort = next(iter(server._verify_cohorts.values()))
        assert cohort.phase == "failed"
        assert all(member.body is None and not member.waiters for member in cohort.members.values())

    async def test_evaluation_cancellation_releases_every_waiter(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        comparison_started = asyncio.Event()
        never_finish = asyncio.Event()

        async def blocked_compare(*args, **kwargs):
            comparison_started.set()
            await never_finish.wait()

        monkeypatch.setattr(server, "_collect_comparisons", blocked_compare)
        waiters = [
            asyncio.create_task(server.verify(self._verify_request(0, task_index=24))),
            asyncio.create_task(server.verify(self._verify_request(1, task_index=24))),
        ]
        await asyncio.wait_for(comparison_started.wait(), timeout=1.0)
        cohort = next(iter(server._verify_cohorts.values()))
        evaluation_task = cohort.evaluation_task
        assert evaluation_task is not None

        evaluation_task.cancel()
        await asyncio.gather(evaluation_task, return_exceptions=True)
        results = await asyncio.gather(*waiters, return_exceptions=True)

        assert all(
            isinstance(result, HTTPException) and result.status_code == 503 and "cancelled" in str(result.detail)
            for result in results
        )
        assert cohort.phase == "failed"
        assert all(member.body is None and not member.waiters for member in cohort.members.values())

    async def test_server_instances_do_not_share_cohorts(self, config):
        config = config.model_copy(update={"num_rollouts_per_prompt": 2})
        first = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        second = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())

        first_waiter = asyncio.create_task(first.verify(self._verify_request(0, task_index=25)))
        await asyncio.sleep(0)

        assert len(first._verify_cohorts) == 1
        assert second._verify_cohorts == {}
        first_waiter.cancel()
        await asyncio.gather(first_waiter, return_exceptions=True)
        cohort = next(iter(first._verify_cohorts.values()))
        assert cohort.phase == "collecting"
        assert cohort.collection_timeout_task is not None

    async def test_terminal_tombstones_are_bounded_and_expire(self, config, monkeypatch: MonkeyPatch):
        config = config.model_copy(
            update={
                "num_rollouts_per_prompt": 2,
                "cohort_result_ttl_s": 3600.0,
                "max_terminal_cohorts": 2,
            }
        )
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        monkeypatch.setattr(server, "_collect_comparisons", AsyncMock(return_value=collected_comparisons([1.0, 2.0])))

        for task_index in (30, 31, 32):
            await asyncio.gather(
                server.verify(self._verify_request(0, task_index=None, group_id=f"group-{task_index}")),
                server.verify(self._verify_request(1, task_index=None, group_id=f"group-{task_index}")),
            )

        server._prune_terminal_cohorts()
        assert len(server._verify_cohorts) == 2

        for cohort in server._verify_cohorts.values():
            cohort.terminal_at = 0.0
        monkeypatch.setattr(resources_servers.genrm_compare.app.time, "monotonic", lambda: 3601.0)
        server._prune_terminal_cohorts()
        assert server._verify_cohorts == {}


class TestFixedBaseline:
    async def test_singleton_verify_calls_judge(self) -> None:
        config = GenRMCompareConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            domain="rlhf",
            name="genrm_compare",
            genrm_model_server=ModelServerRef(type="responses_api_models", name="genrm_model"),
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=1024),
            comparison_mode="fixed_baseline",
            num_rollouts_per_prompt=1,
        )
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        server._run_single_comparison = AsyncMock(
            return_value=(5.0, 1.0, 1.0, 5.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0)
        )
        request = GenRMCompareVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="hello", type="message")],
                metadata={"baseline_response": "baseline"},
            ),
            response=NeMoGymResponse(
                id="rollout",
                created_at=0.0,
                model="dummy_model",
                tools=[],
                parallel_tool_calls=True,
                tool_choice="auto",
                output=[
                    NeMoGymResponseOutputMessage(
                        id="message",
                        role="assistant",
                        status="completed",
                        type="message",
                        content=[NeMoGymResponseOutputText(text="answer", type="output_text", annotations=[])],
                    )
                ],
                object="response",
            ),
            rollout_index=0,
        )

        result = await server.verify(request)

        assert result.reward == 5.0
        server._run_single_comparison.assert_awaited_once()

    async def test_counterbalances_and_keeps_clean_scores(self, monkeypatch: MonkeyPatch) -> None:
        config = GenRMCompareConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            domain="rlhf",
            name="genrm_compare",
            genrm_model_server=ModelServerRef(type="responses_api_models", name="genrm_model"),
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=1024),
            comparison_mode="fixed_baseline",
            score_source="rubric_mean",
            num_judges_per_comparison=2,
        )
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())
        response_objs = [{"output": [{"type": "message", "content": [{"type": "output_text", "text": "rollout"}]}]}]
        seen_pairs = []

        async def compare(_conversation, response_1, response_2, **_kwargs):
            first = response_1["output"][0]["content"][0]["text"]
            second = response_2["output"][0]["content"][0]["text"]
            seen_pairs.append((first, second))
            if first == "rollout":
                return 5.0, 1.0, 1.0, 4.0, 2.0, 2.0, 100.0, 20.0, 0.0, 0.0, 0.0
            return 1.0, 4.0, 6.0, 2.0, 3.0, 5.0, 200.0, 40.0, 1.0, 0.0, 0.0

        monkeypatch.setattr(server, "_run_single_comparison", compare)
        (
            rewards,
            raw_scores,
            clean_scores,
            metrics,
            _,
            overall_raw,
            overall_adjusted,
            adjustments,
        ) = await server._run_fixed_baseline_compare([], response_objs, "baseline", None, "prompt")

        assert set(seen_pairs) == {("rollout", "baseline"), ("baseline", "rollout")}
        assert rewards == approx([4.5])
        assert raw_scores == approx([4.5])
        assert clean_scores == approx([4.5])
        assert overall_raw == approx([3.5])
        assert overall_adjusted == approx([3.5])
        assert adjustments == approx([0.0])
        assert metrics["genrm_parse_failure_rate_per_group"] == 0.0
        assert metrics["genrm_rubric_parse_failure_rate_per_group"] == 0.0
        assert metrics["genrm_input_tokens_per_comparison_mean"] == 150.0
        assert metrics["genrm_input_tokens_per_comparison_p50"] == 150.0
        assert metrics["genrm_input_tokens_per_comparison_p95"] == 195.0
        assert metrics["genrm_output_tokens_per_comparison_mean"] == 30.0
        assert metrics["genrm_output_tokens_per_comparison_p50"] == 30.0
        assert metrics["genrm_output_tokens_per_comparison_p95"] == 39.0
        assert metrics["genrm_output_tokens_total_per_group"] == 60.0
        assert metrics["genrm_max_output_tokens_hit_rate_per_group"] == 0.5
        assert "mean_individual_score" in metrics
        assert "std_individual_score" in metrics
        assert "tiebreak_usage_rate" in metrics

    async def test_failed_rubric_parse_is_neutral_but_not_clean(self, monkeypatch: MonkeyPatch) -> None:
        config = GenRMCompareConfig(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            domain="rlhf",
            name="genrm_compare",
            genrm_model_server=ModelServerRef(type="responses_api_models", name="genrm_model"),
            genrm_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=1024),
            comparison_mode="fixed_baseline",
            score_source="rubric_mean",
        )
        server = GenRMCompareResourcesServer.model_construct(config=config, server_client=MagicMock())

        async def failed(*_args, **_kwargs):
            return 3.0, 3.0, 3.5, 4.0, 2.0, 2.0, -1.0, -1.0, -1.0, 0.0, 1.0

        monkeypatch.setattr(server, "_run_single_comparison", failed)
        response_objs = [{"output": [{"type": "message", "content": [{"type": "output_text", "text": "rollout"}]}]}]
        rewards, _, clean_scores, metrics, *_ = await server._run_fixed_baseline_compare(
            [], response_objs, "baseline", None, "prompt"
        )

        assert rewards == approx([3.0])
        assert clean_scores == [None]
        assert metrics["genrm_parse_failure_rate_per_group"] == 0.0
        assert metrics["genrm_rubric_parse_failure_rate_per_group"] == 1.0


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
        mock_http_response = AsyncMock(ok=True)
        mock_http_response.read = AsyncMock(
            return_value=json.dumps(
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": '{"score_1": 4, "score_2": 2, "ranking": 2}'}],
                        }
                    ]
                }
            ).encode()
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

    def test_rubric_and_overall_scores_are_kept_separately(self):
        server, mock_client = self._make_server()
        server.config.score_source = "rubric_mean"
        answer = {
            "rubric_evaluations": [
                {"rubric_id": 1, "score_1": 5, "score_2": 2, "ranking": 1},
                {"rubric_id": 2, "score_1": 3, "score_2": 4, "ranking": 5},
            ],
            "overall": {"score_1": 2, "score_2": 5, "ranking": 6},
        }
        response = {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]}],
        }
        mock_client.post = AsyncMock(return_value=self._http_response(response))

        result = asyncio.run(
            server._run_single_comparison(
                [],
                self._make_response_obj("one"),
                self._make_response_obj("two"),
                expected_rubric_ids=(1, 2),
            )
        )

        assert result == approx((4.0, 3.0, 3.0, 2.0, 5.0, 6.0, -1.0, -1.0, -1.0, 0.0, 0.0))

    def test_rubric_parse_failure_keeps_valid_overall_diagnostic(self):
        server, _ = self._make_server()
        server.config.score_source = "rubric_mean"
        server.config.genrm_parse_retries = 0

        result = asyncio.run(
            server._run_single_comparison(
                [],
                self._make_response_obj("one"),
                self._make_response_obj("two"),
                expected_rubric_ids=(1,),
            )
        )

        assert result == approx((3.0, 3.0, 3.5, 4.0, 2.0, 2.0, -1.0, -1.0, -1.0, 0.0, 1.0))

    def test_usage_is_recorded(self):
        server, mock_client = self._make_server()
        response = {
            "status": "completed",
            "usage": {"input_tokens": 100, "output_tokens": 25},
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"score_1": 4, "score_2": 2, "ranking": 2}'}],
                }
            ],
        }
        mock_client.post = AsyncMock(return_value=self._http_response(response))

        result = asyncio.run(
            server._run_single_comparison([], self._make_response_obj("one"), self._make_response_obj("two"))
        )

        assert result[6:9] == approx((100.0, 25.0, 0.0))

    def test_rubric_mean_requires_explicit_ids(self):
        server, _ = self._make_server()
        server.config.score_source = "rubric_mean"

        with pytest.raises(ValueError, match="requires expected_rubric_ids"):
            asyncio.run(
                server._run_single_comparison([], self._make_response_obj("one"), self._make_response_obj("two"))
            )

    @staticmethod
    def _http_response(body: dict):
        response = AsyncMock(ok=True)
        response.read = AsyncMock(return_value=json.dumps(body).encode())
        return response

    @staticmethod
    def _incomplete_response():
        """A Responses API object cut off by max_output_tokens: reasoning present, no verdict text."""
        return {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "..."}]}],
        }

    def test_output_budget_exhausted_detection(self):
        assert _output_budget_exhausted(self._incomplete_response())
        assert not _output_budget_exhausted({"status": "completed", "output": []})
        assert not _output_budget_exhausted(
            {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}}
        )
        assert not _output_budget_exhausted({"status": "incomplete"})
        assert not _output_budget_exhausted(None)

    def test_budget_exhaustion_is_named_in_judge_error(self, caplog):
        """Every attempt hits max_output_tokens -> JudgeError says so, with the budget, instead of a generic message."""
        server, mock_client = self._make_server()
        server.config.genrm_parse_retries = 1
        server.config.genrm_parse_retry_sleep_s = 0.0
        mock_client.post = AsyncMock(return_value=self._http_response(self._incomplete_response()))

        with caplog.at_level("WARNING"), pytest.raises(JudgeError) as error:
            asyncio.run(
                server._run_single_comparison(
                    [{"role": "user", "content": "q"}], self._make_response_obj("a"), self._make_response_obj("b")
                )
            )

        assert mock_client.post.await_count == 2
        assert "no completed answer after 2 attempts" in str(error.value)
        assert "2 of them exhausted max_output_tokens=1024" in str(error.value)
        assert sum("output budget exhausted" in r.getMessage() for r in caplog.records) == 2

    def test_budget_exhaustion_then_verdict_recovers(self):
        """One exhausted attempt followed by a completed verdict parses normally."""
        server, mock_client = self._make_server()
        server.config.genrm_parse_retries = 1
        server.config.genrm_parse_retry_sleep_s = 0.0
        completed = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"score_1": 4, "score_2": 2, "ranking": 2}'}],
                }
            ],
        }
        mock_client.post = AsyncMock(
            side_effect=[self._http_response(self._incomplete_response()), self._http_response(completed)]
        )

        result = asyncio.run(
            server._run_single_comparison(
                [{"role": "user", "content": "q"}], self._make_response_obj("a"), self._make_response_obj("b")
            )
        )

        assert result == (4.0, 2.0, 2.0, 4.0, 2.0, 2.0, -1.0, -1.0, -1.0, 0.0, 0.0)

    def test_empty_completed_answer_keeps_generic_judge_error(self):
        """A completed but empty answer is not budget exhaustion; the message stays generic."""
        server, mock_client = self._make_server()
        server.config.genrm_parse_retries = 0
        empty = {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": ""}]}],
        }
        mock_client.post = AsyncMock(return_value=self._http_response(empty))

        with pytest.raises(JudgeError) as error:
            asyncio.run(
                server._run_single_comparison(
                    [{"role": "user", "content": "q"}], self._make_response_obj("a"), self._make_response_obj("b")
                )
            )

        assert str(error.value) == "Judge returned no completed answer after 1 attempts"
