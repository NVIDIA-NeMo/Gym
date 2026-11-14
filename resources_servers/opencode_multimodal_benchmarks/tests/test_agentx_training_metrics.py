# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetricsRequest
from resources_servers.opencode_multimodal_benchmarks.agentx_tool_rewards import (
    AgentXToolRewardConfig,
    AgentXToolRewardServer,
)


@pytest.fixture
def server():
    return AgentXToolRewardServer(
        config=AgentXToolRewardConfig(
            host="127.0.0.1", port=12345, name="agentx", entrypoint="agentx_tool_rewards.py"
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def rows_for(scores, *, adjustment, record_format, repeats=1):
    rows = []
    for task, score in enumerate(scores):
        for repeat in range(repeats):
            auxiliary = adjustment if score > 0 and record_format != "correctness_only" else 0.0
            row = {
                TASK_INDEX_KEY_NAME: task,
                ROLLOUT_INDEX_KEY_NAME: repeat,
                "reward": score + auxiliary,
                "mask_sample": False,
                "verifier_metadata": {
                    "benchmark": "AgentX",
                    "index": str(task),
                    "modality": "Image" if task < 3 else "Video",
                    "expected_samples": len(scores),
                },
            }
            if record_format == "components":
                row["reward_components"] = {"verifier": score, "opencode_tool_reward": auxiliary}
                # Current components must take precedence over legacy evidence.
                row["opencode_tool_reward"] = {"base_reward": 0.2}
            elif record_format == "legacy":
                row["reward_components"] = None
                row["opencode_tool_reward"] = {"base_reward": score}
            rows.append(row)
    return rows


@pytest.mark.parametrize("record_format", ["components", "legacy", "correctness_only"])
async def test_correct_answer_keeps_training_penalty_out_of_accuracy(server, record_format):
    rows = rows_for([1.0], adjustment=-0.05, record_format=record_format)
    original = deepcopy(rows)
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert result.agent_metrics["accuracy"] == result.agent_metrics["AgentX/Image/accuracy"] == 1.0
    assert result.key_metrics["AgentX/accuracy"] == 1.0
    expected_reward = 1.0 if record_format == "correctness_only" else 0.95
    assert result.agent_metrics["mean/reward"] == pytest.approx(expected_reward)
    assert rows == original


@pytest.mark.parametrize("record_format", ["components", "legacy", "correctness_only"])
@pytest.mark.parametrize("adjustment", [-0.05, 0.10])
async def test_partial_credit_reports_correctness_and_preserves_total_reward(server, record_format, adjustment):
    rows = rows_for([0.65, 0, 0, 0, 0], adjustment=adjustment, record_format=record_format, repeats=2)
    original = deepcopy(rows)
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    expected_reward = (0.65 + (0 if record_format == "correctness_only" else adjustment)) / 5
    for metrics in [result.agent_metrics, *result.repeat_level_metrics]:
        assert metrics["accuracy"] == metrics["AgentX/accuracy"] == pytest.approx(0.13)
        assert metrics["AgentX/Image/accuracy"] == pytest.approx(0.65 / 3)
        assert metrics["AgentX/Video/accuracy"] == 0.0
        assert metrics["mean/reward"] == pytest.approx(expected_reward)
    assert len(result.repeat_level_metrics) == 2
    assert result.agent_metrics["AgentX/report"] == [
        {"modality": "Image", "count": 3, "accuracy": pytest.approx(0.65 / 3)},
        {"modality": "Video", "count": 2, "accuracy": 0.0},
    ]
    assert rows == original


@pytest.mark.parametrize("incomplete", ["missing_task", "missing_repeat", "masked_repeat"])
async def test_training_accuracy_preserves_partial_and_masked_repeat_reporting(server, incomplete):
    rows = rows_for([0.65, 0, 0, 0, 0], adjustment=-0.05, record_format="components", repeats=2)
    if incomplete == "missing_task":
        rows = rows[:-2]
    elif incomplete == "missing_repeat":
        rows = rows[:-1]
    else:
        rows[-1]["mask_sample"] = True
    original = deepcopy(rows)
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    metrics = result.agent_metrics
    assert metrics["AgentX/complete"] is False
    assert "accuracy" not in metrics and "AgentX/accuracy" not in result.key_metrics
    assert "AgentX/Image/accuracy" not in metrics and "AgentX/Video/accuracy" not in metrics
    assert metrics["AgentX/partial_accuracy"] == pytest.approx(0.65 / (4 if incomplete == "missing_task" else 5))
    assert metrics["AgentX/Image/partial_accuracy"] == pytest.approx(0.65 / 3)
    assert metrics["AgentX/Video/partial_accuracy"] == 0.0
    assert rows == original
