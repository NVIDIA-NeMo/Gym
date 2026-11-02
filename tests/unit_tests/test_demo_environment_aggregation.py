# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from scripts.demo_environment_aggregation import example_batch, make_demo_environment

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServer
from nemo_gym.config.config_types import AggregateMetricsRequest


async def test_resources_policies_change_accuracy_without_changing_execution_metrics() -> None:
    batch = example_batch()
    pooled = await make_demo_environment("pooled_subtasks").aggregate_metrics(batch)
    macro = await make_demo_environment("mean_tasks").aggregate_metrics(batch)

    assert pooled.agent_metrics["micro/subtask_accuracy"] == pytest.approx(0.1)
    assert macro.agent_metrics["macro/subtask_accuracy"] == pytest.approx(0.5)
    assert pooled.perf_summary == macro.perf_summary
    assert pooled.perf_summary["mean_completion_tokens"] == 500
    assert pooled.perf_summary["total_completion_tokens"] == 1000
    assert pooled.perf_summary["mean_num_turns"] == 2


async def test_masked_results_contribute_execution_cost_but_not_quality() -> None:
    batch = example_batch()
    batch.verify_responses[1]["mask_sample"] = True

    result = await make_demo_environment("pooled_subtasks").aggregate_metrics(batch)

    assert result.agent_metrics["micro/subtask_accuracy"] == 1.0
    assert result.perf_summary["total_completion_tokens"] == 1000


async def test_missing_observations_reduce_coverage_without_inventing_zero_tokens() -> None:
    batch = example_batch()
    del batch.verify_responses[1]["ng_perf"]

    result = await make_demo_environment("pooled_subtasks").aggregate_metrics(batch)

    assert result.agent_metrics["micro/subtask_accuracy"] == pytest.approx(0.1)
    assert result.perf_summary["overall_observability_coverage"] == 0.5
    assert result.perf_summary["mean_completion_tokens"] == 100

    del batch.verify_responses[0]["ng_perf"]
    result = await make_demo_environment("pooled_subtasks").aggregate_metrics(batch)
    assert result.perf_summary is None


async def test_skipping_verification_keeps_performance_without_calling_resources() -> None:
    environment = make_demo_environment("pooled_subtasks", skip_verification=True)
    with pytest.warns(RuntimeWarning, match="skip_verification=True"):
        result = await environment.aggregate_metrics(example_batch())

    assert result.agent_metrics == {}
    assert result.group_level_metrics == []
    assert result.key_metrics == {}
    assert result.perf_summary["total_completion_tokens"] == 1000
    assert environment.server_client.requests == []


async def test_replay_preserves_input_and_resources_payload() -> None:
    environment = make_demo_environment("pooled_subtasks")
    batch = example_batch()
    original = deepcopy(batch.model_dump())

    first = await environment.aggregate_metrics(batch)
    second = await environment.aggregate_metrics(AggregateMetricsRequest.model_validate_json(batch.model_dump_json()))

    assert first == second
    assert batch.model_dump() == original
    assert environment.server_client.requests[0] == batch


async def test_existing_repeat_token_statistics_and_confidence_intervals_are_preserved() -> None:
    environment = make_demo_environment("pooled_subtasks")
    batch = example_batch()
    repeated = deepcopy(batch.verify_responses)
    for row in repeated:
        row["_ng_rollout_index"] = 1
        row["ng_perf"]["completion_tokens"] *= 2
    batch.verify_responses.extend(repeated)

    original = await SingleAgentTurnEnvironmentServer.aggregate_metrics(environment, batch)
    proposed = await environment.aggregate_metrics(batch)

    assert proposed == original
    assert [row["mean_completion_tokens"] for row in proposed.repeat_level_metrics] == [500, 1000]
    assert proposed.agent_metrics["mean_across_repeats/mean_completion_tokens"] == 750
    assert "ci_low_95_across_repeats/mean_completion_tokens" in proposed.agent_metrics
    assert "ci_high_95_across_repeats/mean_completion_tokens" in proposed.agent_metrics


async def test_macro_accuracy_keeps_equal_task_weight_with_uneven_repeats() -> None:
    batch = example_batch()
    repeated_success = deepcopy(batch.verify_responses[0])
    repeated_success["_ng_rollout_index"] = 1
    batch.verify_responses.append(repeated_success)

    with pytest.warns(UserWarning, match="missing rollouts"):
        result = await make_demo_environment("mean_tasks").aggregate_metrics(batch)

    assert result.agent_metrics["macro/subtask_accuracy"] == 0.5
    assert result.perf_summary["total_completion_tokens"] == 1100


async def test_all_masked_or_empty_batch_has_no_quality_estimate() -> None:
    environment = make_demo_environment("pooled_subtasks")
    batch = example_batch()
    for row in batch.verify_responses:
        row["mask_sample"] = True

    result = await environment.aggregate_metrics(batch)
    assert "micro/subtask_accuracy" not in result.agent_metrics
    assert result.perf_summary["total_completion_tokens"] == 1000

    result = await environment.aggregate_metrics(AggregateMetricsRequest(verify_responses=[]))
    assert result.agent_metrics == {}
    assert result.perf_summary is None
