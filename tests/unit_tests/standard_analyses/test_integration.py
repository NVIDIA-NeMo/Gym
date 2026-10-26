# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from unittest.mock import Mock

import pytest

from nemo_gym.health.types import HealthCheckResult, RolloutDigest
from nemo_gym.standard_analyses.inputs import read_records
from nemo_gym.standard_analyses.integration import analyze_collection
from tests.unit_tests.standard_analyses.fixtures import gym_rows


pytestmark = pytest.mark.asyncio


async def test_collection_exports_without_raw_rollout_upload(tmp_path, monkeypatch):
    exporter = Mock()
    monkeypatch.setattr("nemo_gym.exporters._EXPORTERS", [exporter])
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in gym_rows()))
    directory = await analyze_collection(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest) == 1
    assert manifest[0]["namespace"] == "efficiency"
    report = json.loads((directory / manifest[0]["directory"] / "results.json").read_text())
    assert len(report["results"]) == 5
    exporter.export_standard_analyses.assert_called_once()
    exporter.export_rollouts.assert_not_called()


async def test_collection_separates_benchmark_cohorts(tmp_path, monkeypatch):
    exporter = Mock()
    monkeypatch.setattr("nemo_gym.exporters._EXPORTERS", [exporter])
    rows = gym_rows()
    path = tmp_path / "rollouts.jsonl"
    data = []
    for benchmark in ("benchmark_a", "benchmark_b"):
        data.extend({**row, "_ng_environment_server": benchmark} for row in rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in data))
    directory = await analyze_collection(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest) == 2
    assert len({item["namespace"] for item in manifest}) == 2
    for call in exporter.export_standard_analyses.call_args_list:
        assert call.args[0].results[0].coverage["rollouts_total"] == 12
        assert call.kwargs["namespace"].startswith("efficiency/cohorts/")


async def test_empty_collection_records_no_cohorts(tmp_path, monkeypatch):
    monkeypatch.setattr("nemo_gym.exporters._EXPORTERS", [])
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    directory = await analyze_collection(path)
    assert directory is None
    assert not path.with_name("empty_analyses").exists()


@pytest.mark.parametrize("identity", ["taskset", "task_source"])
async def test_collection_separates_datasets_sharing_an_environment(tmp_path, monkeypatch, identity):
    monkeypatch.setattr("nemo_gym.exporters._EXPORTERS", [])
    rows = []
    for dataset in ("dataset_a", "dataset_b"):
        for index, row in enumerate(gym_rows()):
            row["_ng_environment_server"] = "shared_environment"
            if identity == "taskset":
                row["task_id"] = {"taskset": dataset, "task_id": str(index)}
                row["_ng_task_index"] = index
            else:
                row["task_source"] = dataset
            rows.append(row)
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    directory = await analyze_collection(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert {item["cohort"]["benchmark"] for item in manifest} == {"dataset_a", "dataset_b"}
    assert len({item["namespace"] for item in manifest}) == 2


def health_result(rows, tmp_path, verdicts):
    return HealthCheckResult(
        summary={},
        summary_path=tmp_path / "quality_summary.json",
        verdicts_path=tmp_path / "rollout_verdicts.jsonl",
        rollouts=[
            RolloutDigest(
                task_index=row["_ng_task_index"],
                rollout_index=row["_ng_rollout_index"],
                rollout_id=row["ng_trajectory"]["rollout_id"],
                verdict=verdict,
                findings=[],
                unobserved=[],
                capture_observed=True,
            )
            for row, verdict in zip(rows, verdicts, strict=True)
        ],
    )


async def test_health_results_exclude_unhealthy_and_map_unobserved(tmp_path, monkeypatch):
    exporter = Mock()
    monkeypatch.setattr("nemo_gym.exporters._EXPORTERS", [exporter])
    rows = gym_rows()[:3]
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    health = health_result(rows, tmp_path, ["healthy", "unhealthy", "unobserved"])
    await analyze_collection(path, health_result=health)
    report = exporter.export_standard_analyses.call_args.args[0]
    assert report.results[0].coverage["rollouts_excluded_health"] == 1
    assert report.results[0].coverage["rollouts_health_unknown"] == 1
    assert report.results[0].coverage["observations_used"] == 12


async def test_call_less_and_auxiliary_model_rollouts_keep_one_cohort(tmp_path, monkeypatch):
    exporter = Mock()
    monkeypatch.setattr("nemo_gym.exporters._EXPORTERS", [exporter])
    rows = gym_rows()[:3]
    rows[0]["ng_trajectory"]["model_calls"] = []
    rows[0]["ng_model_call_capture"]["calls"] = []
    rows[1]["response"] = {"model": "policy"}
    rows[2]["ng_trajectory"]["model_calls"][0]["response_metadata"]["model"] = "auxiliary"
    # Ownership contradictions are health evidence; the aggregate effort analysis does not rejoin them.
    rows[2]["ng_trajectory"]["invocations"] = [
        {"invocation_id": name, "model_calls": [{"model_call_id": "c0"}]} for name in ("first", "second")
    ]
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    await analyze_collection(path)
    exporter.export_standard_analyses.assert_called_once()
    call = exporter.export_standard_analyses.call_args
    assert call.kwargs["namespace"] == "efficiency"
    assert call.args[0].cohort["model"] == "policy"
    assert call.args[0].results[0].coverage["rollouts_total"] == 3


async def test_shards_without_merge_export_on_caller_thread_and_use_aggregate_scores(tmp_path, monkeypatch):
    exporter = Mock()
    caller = threading.get_ident()
    exporter.export_standard_analyses.side_effect = lambda *args, **kwargs: (
        pytest.fail("Exporter ran on a background thread") if threading.get_ident() != caller else None
    )
    monkeypatch.setattr("nemo_gym.exporters._EXPORTERS", [exporter])
    rows = gym_rows()[:2]
    paths = []
    for i, row in enumerate(rows):
        path = tmp_path / f"shard{i}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        paths.append(path)
    output = tmp_path / "combined.jsonl"
    output.with_name("combined_aggregate_metrics.json").write_text(
        json.dumps(
            [
                {
                    "agent_ref": {"name": "demo-agent"},
                    "agent_metrics": {"pass@1/accuracy": 12.5, "pass@2/accuracy": 25.0},
                }
            ]
        )
    )
    await analyze_collection(output, paths=paths, health_result=health_result(rows, tmp_path, ["healthy"] * 2))
    assert not output.exists()
    report = exporter.export_standard_analyses.call_args.args[0]
    assert report.results[-1].tables["detail"].rows == [[1, "accuracy", 12.5], [2, "accuracy", 25.0]]


async def test_no_observability_does_not_write_or_export_empty_tables(tmp_path, monkeypatch):
    exporter = Mock()
    monkeypatch.setattr("nemo_gym.exporters._EXPORTERS", [exporter])
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1}))
    assert await analyze_collection(path) is None
    assert not path.with_name("rollouts_analyses").exists()
    exporter.export_standard_analyses.assert_not_called()


async def test_analysis_failure_is_guarded_and_does_not_export(tmp_path, monkeypatch, caplog):
    exporter = Mock()
    monkeypatch.setattr("nemo_gym.exporters._EXPORTERS", [exporter])
    path = tmp_path / "rollouts.jsonl"
    path.write_text("invalid json")
    assert await analyze_collection(path) is None
    exporter.export_standard_analyses.assert_not_called()
    assert "Standard analyses failed" in caplog.text


async def test_health_row_mismatch_is_rejected(tmp_path):
    rows = gym_rows()[:1]
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(rows[0]))
    health = health_result(rows, tmp_path, ["healthy"])
    health.rollouts[0].task_index = 999
    with pytest.raises(ValueError, match="does not match"):
        read_records([path], run_id="run", health_result=health)


@pytest.mark.parametrize("merge", [True, False])
async def test_aggregation_runs_analyses_after_health_with_shard_identity(tmp_path, monkeypatch, merge):
    from nemo_gym.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper, RolloutCollectionHelper

    rows = gym_rows()[:2]
    shard = tmp_path / "shard.jsonl"
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "combined.jsonl"
    health = health_result(rows, tmp_path, ["healthy", "unhealthy"])
    events = []

    async def aggregate(self, results, rows, output_fpath, **kwargs):
        events.append("aggregate")
        path = output_fpath.with_name(output_fpath.stem + "_aggregate_metrics.json")
        path.write_text("[]")
        return path

    def check_health(paths, **kwargs):
        assert paths == [shard]
        events.append("health")
        return health

    async def analyses(path, **kwargs):
        events.append("analyses")
        assert path == output
        assert kwargs == {"paths": [shard], "health_result": health}

    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", aggregate)
    monkeypatch.setattr("nemo_gym.rollout_health.run_health_checks", check_health)
    monkeypatch.setattr("nemo_gym.rollout_health.format_health_report", lambda result: "checked")
    monkeypatch.setattr("nemo_gym.standard_analyses.integration.analyze_collection", analyses)
    monkeypatch.setattr("nemo_gym.exporters._EXPORTERS", [])
    await RolloutAggregationHelper().run_from_config(
        RolloutAggregationConfig(
            input_glob=str(shard),
            output_jsonl_fpath=str(output),
            merge_shards=merge,
        )
    )
    assert events == ["aggregate", "health", "analyses"]
    assert output.exists() is merge


async def test_multiple_cohorts_do_not_relabel_whole_run_pass_metrics(tmp_path, monkeypatch):
    exporter = Mock()
    monkeypatch.setattr("nemo_gym.exporters._EXPORTERS", [exporter])
    rows = gym_rows()[:2]
    for i, row in enumerate(rows):
        row["task_source"] = f"dataset-{i}"
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.with_name("rollouts_aggregate_metrics.json").write_text(
        json.dumps(
            [
                {
                    "agent_ref": {"name": "demo-agent"},
                    "agent_metrics": {"pass@1/accuracy": 50.0},
                }
            ]
        )
    )
    await analyze_collection(path)
    assert exporter.export_standard_analyses.call_count == 2
    for call in exporter.export_standard_analyses.call_args_list:
        assert call.args[0].results[-1].status == "unavailable"
