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
"""A strict run that ends with a bounded share of unsettled rollouts writes a partial score (max_failed_rollout_fraction)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import orjson
import pytest

import nemo_gym.rollouts.rollout_collection as collection
from nemo_gym.artifacts.run_files import PARTIAL_SCORE_KEY
from nemo_gym.metrics.reward_profile import compute_aggregate_metrics
from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollouts.rollout_records import coverage_path_for
from nemo_gym.rollouts.rollout_recovery import IncompleteEvaluationError
from nemo_gym.utils.path_utils import aggregate_metrics_path_for, partial_aggregate_metrics_path_for
from tests.unit_tests.rollouts.test_rollout_collection import FakeResponse, failing_row, install_fake_server_client


TASKS = 20
FAILED_TASK = 7
SCORED = {"reward": 1.0, "response": {}}


def infrastructure_failure(failure_class: str = "infrastructure_error") -> dict:
    return {"_ng_failure_class": failure_class, "failure_kind": "verifier_error", "response": {}}


# A result the environment server masked because its provider failed: saved, and run again by a resume.
MASKED = {"response": {}, "reward": 0.0, "mask_sample": True, "failure_kind": "provider_unavailable"}


@pytest.fixture
def strict_config(tmp_path, monkeypatch) -> RolloutCollectionConfig:
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(failing_row(task)) + "\n" for task in range(TASKS)))
    return RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        route_failures_to_sidecar=True,
        require_complete=True,
        disable_health_check=True,
        num_samples_in_parallel=1,
        dispatch_reservation_interval_s=0.0,
    )


def install_server(monkeypatch, replies: dict[int, dict], *, aggregate_status: int = 200) -> list[str]:
    """Every rollout scores 1.0 except the tasks in ``replies``; returns the paths the run posted to."""
    urls = []

    async def post(**kwargs):
        urls.append(kwargs["url_path"])
        if kwargs["url_path"] == "/aggregate_metrics":
            if aggregate_status != 200:
                return FakeResponse(aggregate_status, {"detail": "metric hook failed"})
            rows = [dict(r) for r in kwargs["json"].verify_responses]
            return FakeResponse(200, compute_aggregate_metrics(rows).model_dump())
        # A copy per call: collection stamps the rollout's identity onto the result it receives.
        return FakeResponse(200, dict(replies.get(kwargs["json"].get("_ng_task_index"), SCORED)))

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    return urls


def record_exports(monkeypatch) -> list[dict]:
    exported: list[dict] = []
    monkeypatch.setattr(collection, "export_metrics", lambda metrics, **_: exported.append(metrics))
    monkeypatch.setattr(collection, "get_exporters", lambda: [object()])
    return exported


async def test_a_failed_share_within_the_limit_writes_a_partial_score_a_resume_replaces(strict_config, monkeypatch):
    strict_config.max_failed_rollout_fraction = 0.1
    install_server(monkeypatch, {FAILED_TASK: infrastructure_failure()})
    exported = record_exports(monkeypatch)
    output = Path(strict_config.output_jsonl_fpath)

    # The run is still incomplete, so a job chain retries it as before.
    with pytest.raises(IncompleteEvaluationError, match=f"{TASKS - 1}/{TASKS} samples completed"):
        await RolloutCollectionHelper().run_from_config(strict_config)

    record = {
        "accepted_failures": 1,
        "counted_as_zero": 1,
        "masked": 0,
        "expected": TASKS,
        "accepted_failure_fraction": 1 / TASKS,
        "max_failed_rollout_fraction": 0.1,
    }
    # The partial score has its own file; readers of the aggregate see no score, and no tracker receives it.
    assert not aggregate_metrics_path_for(output).exists()
    entries = orjson.loads(partial_aggregate_metrics_path_for(output).read_bytes())
    assert [entry[PARTIAL_SCORE_KEY] for entry in entries] == [record]
    # The failed rollout is in the denominator as a zero.
    assert entries[0]["agent_metrics"]["mean/reward"] == pytest.approx((TASKS - 1) / TASKS)
    assert not any("mean/reward" in key for metrics in exported for key in metrics)
    assert [metrics["coverage/accepted_failures"] for metrics in exported if "coverage/scored" in metrics] == [1]
    coverage = json.loads(coverage_path_for(output).read_text())
    assert coverage[PARTIAL_SCORE_KEY] == record
    assert (coverage["scored"], coverage["complete"]) == (TASKS, False)

    # The segment that settles the rollout writes the aggregate and removes the partial score.
    strict_config.resume_from_cache = True
    install_server(monkeypatch, {})
    await RolloutCollectionHelper().run_from_config(strict_config)

    assert not partial_aggregate_metrics_path_for(output).exists()
    (entry,) = orjson.loads(aggregate_metrics_path_for(output).read_bytes())
    assert PARTIAL_SCORE_KEY not in entry and entry["agent_metrics"]["mean/reward"] == 1.0
    assert PARTIAL_SCORE_KEY not in json.loads(coverage_path_for(output).read_text())


async def test_failed_masked_and_missing_rollouts_each_count_once(strict_config, monkeypatch):
    """One listed failure class, one other failure, one masked result and one never-started rollout."""
    listed, other, masked, missing = 3, 5, 9, 11
    strict_config.max_failed_rollout_fraction = 0.25
    strict_config.count_failure_classes_as_zero = ["infrastructure_error"]
    urls = install_server(
        monkeypatch, {listed: infrastructure_failure(), other: infrastructure_failure("timeout"), masked: MASKED}
    )
    verified: list[list[dict]] = []
    aggregate = collection.RolloutCollectionHelper._aggregate_entries

    async def recording_aggregate(self, results, rows, **kwargs):
        verified.append(results)
        return await aggregate(self, results, rows, **kwargs)

    monkeypatch.setattr(collection.RolloutCollectionHelper, "_aggregate_entries", recording_aggregate)

    class DropsOneRollout(RolloutCollectionHelper):
        def _run_examples_with_metadata(self, examples, *args, **kwargs):
            kept = [row for row in examples if row["_ng_task_index"] != missing]
            return super()._run_examples_with_metadata(kept, *args, **kwargs)

    with pytest.raises(IncompleteEvaluationError):
        await DropsOneRollout().run_from_config(strict_config)

    assert "/aggregate_metrics" in urls
    output = Path(strict_config.output_jsonl_fpath)
    (entry,) = orjson.loads(partial_aggregate_metrics_path_for(output).read_bytes())
    assert entry[PARTIAL_SCORE_KEY] == {
        "accepted_failures": 4,
        "counted_as_zero": 3,
        "masked": 1,
        "expected": TASKS,
        "accepted_failure_fraction": 4 / TASKS,
        "max_failed_rollout_fraction": 0.25,
    }
    # Each rollout enters the metric input once: the listed failure is not counted again by the partial score.
    (rows,) = verified
    tasks = sorted(row["_ng_task_index"] for row in rows)
    assert tasks == list(range(TASKS))
    zeros = sorted(row["_ng_task_index"] for row in rows if row["reward"] == 0.0 and not row.get("mask_sample"))
    assert zeros == [listed, other, missing]
    coverage = json.loads(coverage_path_for(output).read_text())
    assert (coverage["masked_retryable"], coverage["not_started"]) == (1, 1)


@pytest.mark.parametrize("limit", [None, 0.01], ids=["unset", "above-limit"])
async def test_without_the_option_or_above_the_limit_no_score_is_written(strict_config, monkeypatch, limit):
    strict_config.max_failed_rollout_fraction = limit
    output = Path(strict_config.output_jsonl_fpath)
    # An earlier segment's aggregate and partial score describe other rows.
    aggregate_metrics_path_for(output).write_text("[]")
    partial_aggregate_metrics_path_for(output).write_text("[]")
    urls = install_server(monkeypatch, {FAILED_TASK: infrastructure_failure()})

    with pytest.raises(IncompleteEvaluationError):
        await RolloutCollectionHelper().run_from_config(strict_config)

    assert "/aggregate_metrics" not in urls
    assert not aggregate_metrics_path_for(output).exists()
    assert not partial_aggregate_metrics_path_for(output).exists()
    assert PARTIAL_SCORE_KEY not in json.loads(coverage_path_for(output).read_text())


async def test_a_metric_hook_that_fails_on_the_zero_rows_leaves_no_score(strict_config, monkeypatch):
    strict_config.max_failed_rollout_fraction = 0.1
    urls = install_server(monkeypatch, {FAILED_TASK: infrastructure_failure()}, aggregate_status=500)

    # The run's own rejection stands, not an aggregation error, so the chain decides its retry as before.
    with pytest.raises(IncompleteEvaluationError):
        await RolloutCollectionHelper().run_from_config(strict_config)

    output = Path(strict_config.output_jsonl_fpath)
    assert "/aggregate_metrics" in urls
    assert not aggregate_metrics_path_for(output).exists()
    assert not partial_aggregate_metrics_path_for(output).exists()
    coverage = json.loads(coverage_path_for(output).read_text())
    assert PARTIAL_SCORE_KEY not in coverage and coverage["scored"] == TASKS - 1
