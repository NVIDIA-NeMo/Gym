# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetricsRequest
from nemo_gym.metrics import compute_aggregate_metrics
from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksConfig, VisualBenchmarksServer
from resources_servers.opencode_multimodal_benchmarks.metrics import average_repeats


@pytest.fixture(params=["VStarBench", "HRBench8K", "VTCBench"])
def repeated_benchmark(request):
    benchmark = request.param
    config_type, server_type = (VisualBenchmarksConfig, VisualBenchmarksServer)
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
            },
        }
        for task in range(3)
        for repeat in range(2)
    ]
    return benchmark, server, rows


def planned_request(rows):
    inventory = {}
    for row in rows:
        inventory.setdefault(row[TASK_INDEX_KEY_NAME], []).append(row[ROLLOUT_INDEX_KEY_NAME])
    return AggregateMetricsRequest(verify_responses=rows, expected_rollout_indices=inventory)


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


def test_sample_repeat_guard_is_independent_of_framework_validation(repeated_benchmark):
    _, _, rows = repeated_benchmark
    with pytest.raises(ValueError, match="^Duplicate sample repeat$"):
        average_repeats([[rows[0], deepcopy(rows[0])]])


def test_repeat_metadata_must_be_identical(repeated_benchmark):
    _, _, rows = repeated_benchmark
    changed = deepcopy(rows[1])
    changed["verifier_metadata"]["category"] = "changed"
    with pytest.raises(ValueError, match="^Inconsistent metadata across sample repeats$"):
        average_repeats([[rows[0], changed]])


@pytest.mark.asyncio
async def test_uniformly_absent_repeat_cannot_publish_headline(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    planned = planned_request(rows)
    survivors = [row for row in rows if row[ROLLOUT_INDEX_KEY_NAME] == 0]
    result = await server.aggregate_metrics(planned.model_copy(update={"verify_responses": survivors}))
    assert result.agent_metrics[f"{benchmark}/complete"] is False
    assert f"{benchmark}/partial_accuracy" in result.agent_metrics
    assert result.key_metrics == {}
    recovered = await server.aggregate_metrics(planned)
    assert recovered.agent_metrics[f"{benchmark}/complete"] is True
    assert recovered.agent_metrics["accuracy"] == pytest.approx(0.25 if benchmark == "HRBench8K" else 1 / 3)


@pytest.mark.asyncio
async def test_unknown_attempt_plan_cannot_establish_completeness(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows))
    assert result.agent_metrics[f"{benchmark}/complete"] is False
    assert result.key_metrics == {}


@pytest.mark.asyncio
async def test_collection_sidecars_preserve_planned_repeat_denominator(repeated_benchmark, tmp_path, monkeypatch):
    from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper
    from tests.unit_tests.rollouts.test_rollout_collection import (
        FakeResponse,
        install_fake_server_client,
        typed_judge_failure_result,
    )

    benchmark, server, rows = repeated_benchmark
    source = tmp_path / "input.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "agent_ref": {"name": "my_agent"},
                    "responses_create_params": {"input": []},
                    "verifier_metadata": row["verifier_metadata"],
                }
            )
            + "\n"
            for row in rows
            if row[ROLLOUT_INDEX_KEY_NAME] == 0
        )
    )
    requests = []

    async def post(server_name, url_path, json, **kwargs):
        if url_path == "/run":
            result = (
                typed_judge_failure_result()
                if json[ROLLOUT_INDEX_KEY_NAME] == 1
                else {"reward": float(json[TASK_INDEX_KEY_NAME] < 2), "response": {}}
            )
            return FakeResponse(200, result | {"verifier_metadata": json["verifier_metadata"]})
        assert url_path == "/aggregate_metrics"
        requests.append(json)
        return FakeResponse(200, (await server.aggregate_metrics(json)).model_dump())

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    monkeypatch.setattr(
        "nemo_gym.rollouts.rollout_collection.get_global_config_dict", lambda: client.global_config_dict
    )
    output = tmp_path / "output.jsonl"
    await RolloutCollectionHelper().run_from_config(
        RolloutCollectionConfig(
            input_jsonl_fpath=str(source),
            output_jsonl_fpath=str(output),
            num_repeats=2,
            route_failures_to_sidecar=True,
            disable_health_check=True,
        )
    )
    assert len(output.read_text().splitlines()) == 3
    failures = [json.loads(line) for line in (tmp_path / "output_failures.jsonl").read_text().splitlines()]
    assert len(failures) == 3 and all(row[ROLLOUT_INDEX_KEY_NAME] == 1 for row in failures)
    request = requests[-1]
    assert request.expected_rollout_indices == {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    metrics = await server.aggregate_metrics(request)
    assert metrics.agent_metrics[f"{benchmark}/complete"] is False
    assert metrics.key_metrics == {}


@pytest.mark.asyncio
async def test_uniformly_masked_repeat_is_partial_through_aggregation_handler(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    for row in rows:
        row["mask_sample"] = row[ROLLOUT_INDEX_KEY_NAME] == 1
    result = await server.aggregate_metrics(planned_request(rows))
    expected = 0.5 if benchmark == "HRBench8K" else 2 / 3
    assert result.agent_metrics[f"{benchmark}/complete"] is False
    assert result.agent_metrics[f"{benchmark}/partial_accuracy"] == pytest.approx(expected)
    assert result.agent_metrics["coverage/masked_rollouts"] == 3
    assert result.agent_metrics["num_repeats"] == 2
    assert "accuracy" not in result.agent_metrics
    assert f"{benchmark}/accuracy" not in result.agent_metrics
    assert f"{benchmark}/accuracy" not in result.key_metrics

    # Rejudging the failed attempts restores the full denominator and headline.
    for row in rows:
        row["mask_sample"] = False
    recovered = await server.aggregate_metrics(planned_request(rows))
    assert recovered.agent_metrics[f"{benchmark}/complete"] is True
    assert recovered.agent_metrics["accuracy"] == pytest.approx(expected / 2)


def test_direct_metrics_exclude_masked_and_unfinished_measurements(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    for row in rows:
        row["mask_sample"] = row[ROLLOUT_INDEX_KEY_NAME] == 1
        if row["mask_sample"]:
            row["reward"] = -100  # An invalid measurement must never enter a quality mean.
    tasks = [[row for row in rows if row[TASK_INDEX_KEY_NAME] == index] for index in range(3)]
    expected = 0.5 if benchmark == "HRBench8K" else 2 / 3
    for invalid_flag in ("mask_sample", "evaluation_completed"):
        if invalid_flag == "evaluation_completed":
            for row in rows:
                row["evaluation_completed"] = not row.pop("mask_sample")
        metrics = server.compute_metrics(tasks)
        assert metrics[f"{benchmark}/complete"] is False
        assert metrics[f"{benchmark}/partial_accuracy"] == pytest.approx(expected)
        assert "accuracy" not in metrics


@pytest.mark.asyncio
async def test_masks_do_not_make_an_unaffected_benchmark_partial(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    other = "VTCBench" if benchmark != "VTCBench" else "VStarBench"
    additional = deepcopy(rows)
    for row in additional:
        row[TASK_INDEX_KEY_NAME] += 3
        row["verifier_metadata"]["benchmark"] = other
    for row in rows:
        row["mask_sample"] = row[ROLLOUT_INDEX_KEY_NAME] == 1
    result = await server.aggregate_metrics(planned_request(rows + additional))
    assert result.agent_metrics[f"{benchmark}/complete"] is False
    assert result.agent_metrics[f"{other}/complete"] is True
    assert f"{benchmark}/accuracy" not in result.key_metrics
    assert f"{other}/accuracy" in result.key_metrics


@pytest.mark.asyncio
async def test_zerobench_extra_masked_attempt_does_not_certify_pass_at_five(repeated_benchmark):
    _, server, _ = repeated_benchmark
    rows = [
        {
            TASK_INDEX_KEY_NAME: task,
            ROLLOUT_INDEX_KEY_NAME: repeat,
            "reward": 1,
            "mask_sample": repeat == 5,
            "verifier_metadata": {"benchmark": "ZeroBench", "index": str(task), "expected_samples": 2},
        }
        for task in range(2)
        for repeat in range(6)
    ]
    result = await server.aggregate_metrics(planned_request(rows))
    assert result.agent_metrics["ZeroBench/complete"] is False
    assert result.agent_metrics["ZeroBench/partial_pass@5"] == 1
    assert "accuracy" not in result.agent_metrics
    assert "ZeroBench/pass@5" not in result.key_metrics


def test_all_invalid_measurements_have_no_quality_score(repeated_benchmark):
    benchmark, server, rows = repeated_benchmark
    for row in rows:
        row["mask_sample"] = True
    tasks = [[row for row in rows if row[TASK_INDEX_KEY_NAME] == index] for index in range(3)]
    metrics = server.compute_metrics(tasks)
    assert metrics[f"{benchmark}/complete"] is False
    assert metrics[f"{benchmark}/completed"] == 0
    assert "accuracy" not in metrics and f"{benchmark}/partial_accuracy" not in metrics


def test_http_aggregation_keeps_masked_repeat_identities(repeated_benchmark):
    from fastapi.testclient import TestClient

    benchmark, server, rows = repeated_benchmark
    for row in rows:
        row["mask_sample"] = row[ROLLOUT_INDEX_KEY_NAME] == 1
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/aggregate_metrics", json={"verify_responses": rows})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["agent_metrics"][f"{benchmark}/complete"] is False
    assert "accuracy" not in result["agent_metrics"]
    assert f"{benchmark}/accuracy" not in result["key_metrics"]
