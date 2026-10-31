# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from types import SimpleNamespace

import pytest

from benchmarks.agentic_if_realworld import rollout_driver
from nemo_gym.rollout_collection import RolloutCollectionHelper
from resources_servers.agentic_if_realworld.benchmark.core import judge, write


TASKS = [f"task-{i}" for i in range(6)]


class FakeDataset:
    synthetic = True
    pin = {"policy_assets": {}}
    report = {}

    def __init__(self, directory, synthetic):
        pass

    def prepared(self):
        return [{"task_id": i, "verifier_metadata": {}, "responses_create_params": {}} for i in TASKS]


def collect(
    tmp_path, monkeypatch, seen, *, parallel, failing=None, failing_write=None, delays=None, judge_override=None
):
    seen.update(active=0, peak=0, started=[], finished=[])

    async def run_cell(dataset, identifier, *, directory, **_):
        seen["started"].append(identifier)
        seen["active"] += 1
        seen["peak"] = max(seen["peak"], seen["active"])
        await asyncio.sleep((delays or {}).get(identifier, 0.01))
        seen["active"] -= 1
        if identifier == failing:
            raise ValueError("Ambiguous upstream dispatch")
        seen["finished"].append(identifier)
        write(directory / "cells" / (judge.sha(identifier) + ".json"), {"attempts": [{}]})
        return {"item_id": identifier, "scoring_status": "scored"}

    def write_reports(output, results, dataset):
        if results and results[-1]["item_id"] == failing_write:
            raise OSError("Disk quota exceeded")

    async def write_aggregate_metrics(output, records, client):
        seen["aggregated"] = [row["task_id"] for row in records]

    monkeypatch.setattr(rollout_driver, "Dataset", FakeDataset)
    monkeypatch.setattr(
        rollout_driver, "indexed", lambda path, key: {r["task_id"]: r for r in FakeDataset.prepared(None)}
    )
    monkeypatch.setattr(rollout_driver, "run_cell", run_cell)
    monkeypatch.setattr(rollout_driver, "write_reports", write_reports)
    monkeypatch.setattr(rollout_driver, "write_aggregate_metrics", write_aggregate_metrics)
    monkeypatch.setattr(RolloutCollectionHelper, "setup_server_client", lambda self: None)
    monkeypatch.setattr(RolloutCollectionHelper, "preprocess_examples", lambda self, rows, **_: rows)
    config = SimpleNamespace(
        num_repeats=1,
        fan_out=None,
        responses_create_params=None,
        input_jsonl_fpath=str(tmp_path / "input.jsonl"),
        output_jsonl_fpath=str(tmp_path / "out" / "rollouts.jsonl"),
        limit=None,
        resume_from_cache=False,
        num_samples_in_parallel=parallel,
    )
    resource = {"dataset_dir": "", "synthetic": True, "receipt_dir": str(tmp_path / "receipts")}
    monkeypatch.setenv("SYNTHETIC", "synthetic")
    resource |= {
        "judge_endpoint": "http://synthetic.invalid",
        "judge_model": "synthetic-judge",
        "judge_api_key_env": "SYNTHETIC",
        "judge_reasoning": "medium",
    }
    resource.update(judge_override or {})
    global_config = {
        "agentic_if_realworld_resources": {"resources_servers": {"agentic_if_realworld": resource}},
        "policy_model": {
            "responses_api_models": {"agentic_if_text": {"endpoint": "", "model": "", "api_key_env": ""}}
        },
    }
    asyncio.run(rollout_driver.run_rollout_collection(config, global_config))
    return tmp_path / "out" / "rollouts.jsonl"


def test_tasks_run_concurrently_up_to_the_limit_with_one_row_each(tmp_path, monkeypatch):
    seen = {}
    output = collect(tmp_path, monkeypatch, seen, parallel=3)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert seen["peak"] == 3
    assert sorted(r["task_id"] for r in rows) == TASKS
    assert sorted(seen["aggregated"]) == TASKS


def test_unset_parallelism_stays_sequential(tmp_path, monkeypatch):
    seen = {}
    collect(tmp_path, monkeypatch, seen, parallel=None)
    assert seen["peak"] == 1


def test_failure_stops_admission_and_lets_started_tasks_settle(tmp_path, monkeypatch):
    seen = {}
    with pytest.raises(ValueError, match="Ambiguous upstream dispatch"):
        collect(tmp_path, monkeypatch, seen, parallel=2, failing="task-0", delays={"task-0": 0, "task-1": 0.05})
    # task-1 started before task-0 failed and still finishes; nothing starts after the failure.
    assert seen["started"] == ["task-0", "task-1"]
    assert seen["finished"] == ["task-1"]
    assert seen["aggregated"] == ["task-1"]


def test_write_failure_lets_started_tasks_settle(tmp_path, monkeypatch):
    seen = {}
    with pytest.raises(OSError, match="quota"):
        collect(tmp_path, monkeypatch, seen, parallel=2, failing_write="task-0", delays={"task-0": 0, "task-1": 0.05})
    # task-1 was generating when task-0's row failed to write; it still finishes.
    assert seen["finished"] == ["task-0", "task-1"]
    assert "aggregated" not in seen


@pytest.mark.parametrize(
    "judge_override",
    [{"judge_endpoint": ""}, {"judge_model": ""}, {"judge_api_key_env": "UNSET_SYNTHETIC_JUDGE_KEY"}],
)
def test_invalid_judge_config_fails_before_policy_dispatch(tmp_path, monkeypatch, judge_override):
    monkeypatch.delenv("UNSET_SYNTHETIC_JUDGE_KEY", raising=False)
    seen = {}
    with pytest.raises(ValueError):
        collect(tmp_path, monkeypatch, seen, parallel=1, judge_override=judge_override)
    assert seen["started"] == []
    assert not (tmp_path / "receipts").exists()
