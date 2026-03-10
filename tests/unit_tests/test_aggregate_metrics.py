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
from unittest.mock import MagicMock

import pytest

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseResourcesServerConfig,
    SimpleResourcesServer,
)
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.server_utils import ServerClient


class _TestResourcesServer(SimpleResourcesServer):
    async def verify(self, body):
        pass


def _make_server():
    config = BaseResourcesServerConfig(host="127.0.0.1", port=12345, entrypoint="app.py", name="test_server")
    return _TestResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_verify_responses(tasks, rollouts_per_task, reward_fn=None):
    if reward_fn is None:
        reward_fn = lambda t, r: float((t + r) % 2)

    responses = []
    for task_idx in range(tasks):
        for rollout_idx in range(rollouts_per_task):
            responses.append(
                {
                    TASK_INDEX_KEY_NAME: task_idx,
                    ROLLOUT_INDEX_KEY_NAME: rollout_idx,
                    "reward": reward_fn(task_idx, rollout_idx),
                }
            )
    return responses


class TestAggregateMetricsRoute:
    @pytest.mark.asyncio
    async def test_basic_route(self) -> None:
        server = _make_server()
        responses = _make_verify_responses(tasks=2, rollouts_per_task=4)
        body = AggregateMetricsRequest(verify_responses=responses)

        result = await server.aggregate_metrics(body)

        assert isinstance(result, AggregateMetrics)
        assert len(result.group_level_metrics) == 2
        # Agent metrics should have reward stats
        assert "mean/reward" in result.agent_metrics

    @pytest.mark.asyncio
    async def test_group_level_has_reward_stats(self) -> None:
        server = _make_server()
        responses = _make_verify_responses(tasks=2, rollouts_per_task=3)
        body = AggregateMetricsRequest(verify_responses=responses)

        result = await server.aggregate_metrics(body)

        assert len(result.group_level_metrics) == 2
        group0 = result.group_level_metrics[0]
        assert "mean/reward" in group0

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        server = _make_server()
        body = AggregateMetricsRequest(verify_responses=[])

        result = await server.aggregate_metrics(body)

        assert result.group_level_metrics == []
        assert result.agent_metrics == {}

    @pytest.mark.asyncio
    async def test_agent_metrics_has_overall_stats(self) -> None:
        server = _make_server()
        responses = _make_verify_responses(tasks=3, rollouts_per_task=5, reward_fn=lambda t, r: 1.0)
        body = AggregateMetricsRequest(verify_responses=responses)

        result = await server.aggregate_metrics(body)

        assert result.agent_metrics["mean/reward"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_key_metrics_default(self) -> None:
        server = _make_server()
        responses = _make_verify_responses(tasks=2, rollouts_per_task=3, reward_fn=lambda t, r: 1.0)
        body = AggregateMetricsRequest(verify_responses=responses)

        result = await server.aggregate_metrics(body)

        assert "mean/reward" in result.key_metrics
        assert result.key_metrics["mean/reward"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_histograms_stripped(self) -> None:
        """RewardProfiler produces histograms; they should be stripped from the response."""
        server = _make_server()
        responses = _make_verify_responses(tasks=2, rollouts_per_task=3)
        body = AggregateMetricsRequest(verify_responses=responses)

        result = await server.aggregate_metrics(body)

        for group in result.group_level_metrics:
            assert not any(k.startswith("histogram") for k in group), f"Histogram key found in group: {group.keys()}"
        assert not any(k.startswith("histogram") for k in result.agent_metrics)


class TestDefaultAgentAggregateMetrics:
    @pytest.mark.asyncio
    async def test_default_fallback(self) -> None:
        """Base agent uses the same RewardProfiler logic as the resources server."""
        from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent

        class TestAgent(SimpleResponsesAPIAgent):
            async def responses(self, body=None):
                pass

            async def run(self, body=None):
                pass

        config = BaseResponsesAPIAgentConfig(host="127.0.0.1", port=12345, entrypoint="app.py", name="test_agent")
        agent = TestAgent(config=config, server_client=MagicMock(spec=ServerClient))

        responses = _make_verify_responses(tasks=2, rollouts_per_task=3, reward_fn=lambda t, r: 1.0)
        body = AggregateMetricsRequest(verify_responses=responses)
        result = await agent.aggregate_metrics(body)

        assert isinstance(result, AggregateMetrics)
        assert result.agent_metrics["mean/reward"] == pytest.approx(1.0)
        assert len(result.group_level_metrics) == 2
        assert "mean/reward" in result.key_metrics
