# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import ServerClient
from resources_servers.opencode_multimodal_benchmarks.agentx import AgentXConfig, AgentXServer
from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksConfig, VisualBenchmarksServer


@pytest.fixture(params=["VStarBench", "HRBench8K", "VTCBench", "BabyVision", "AgentX_GoalAcc"])
def repeated_benchmark(request):
    benchmark = request.param
    config_type, server_type = (
        (AgentXConfig, AgentXServer)
        if benchmark == "AgentX_GoalAcc"
        else (VisualBenchmarksConfig, VisualBenchmarksServer)
    )
    server = server_type(
        config=config_type(host="127.0.0.1", port=12345, name="benchmark", entrypoint="app.py"),
        server_client=MagicMock(spec=ServerClient),
    )
    rows = [
        {
            TASK_INDEX_KEY_NAME: task,
            ROLLOUT_INDEX_KEY_NAME: repeat,
            "reward": float(repeat == 0 and task < 2),
            "verifier_metadata": {
                "benchmark": benchmark,
                "index": str(task),
                "expected_samples": 3,
                "cycle_category": "1" if task < 2 else "2",
                "category": "spatial",
                "l2_category": "detail",
                "split": "test",
                "modality": "Image" if task < 2 else "Video",
            },
        }
        for task in range(3)
        for repeat in range(2)
    ]
    return benchmark, server, rows


def aggregate(server, rows):
    return compute_aggregate_metrics(
        rows,
        compute_metrics_fn=server.compute_metrics,
        get_key_metrics_fn=server.get_key_metrics,
        compute_repeat_metrics_fn=server.compute_repeat_metrics,
    )


def test_complete_repeats_have_headline_and_per_repeat_scores(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    result = aggregate(server, rows)
    expected = 0.25 if benchmark == "HRBench8K" else 1 / 3
    assert result.agent_metrics["accuracy"] == pytest.approx(expected)
    assert result.agent_metrics[f"{benchmark}/completed"] == 3
    assert result.agent_metrics[f"{benchmark}/complete"] is True
    scores = {row[ROLLOUT_INDEX_KEY_NAME]: row["accuracy"] for row in result.repeat_level_metrics}
    assert scores == {0: pytest.approx(2 * expected), 1: 0}


def test_missing_repeat_is_partial_even_when_all_tasks_are_present(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    result = aggregate(server, rows[:-1])
    assert result.agent_metrics[f"{benchmark}/complete"] is False
    assert "accuracy" not in result.agent_metrics
    assert f"{benchmark}/partial_accuracy" in result.agent_metrics
    assert f"{benchmark}/accuracy" not in result.agent_metrics


def test_duplicate_measurement_is_rejected(repeated_benchmark):
    _, server, rows = repeated_benchmark
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(server, rows + rows[:1])


def test_duplicate_sample_identity_is_rejected(repeated_benchmark):
    _, server, rows = repeated_benchmark
    duplicate = deepcopy(rows[:2])
    for row in duplicate:
        row[TASK_INDEX_KEY_NAME] = 3
    with pytest.raises(ValueError, match="Duplicate benchmark sample ID"):
        aggregate(server, rows + duplicate)
