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
from nemo_gym.utils.path_utils import aggregate_metrics_path_for
from tests.unit_tests.rollouts.test_rollout_collection import FakeResponse, failing_row, install_fake_server_client


TASKS = 20
FAILED_TASK = 7


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


def install_one_failing_rollout(monkeypatch, *, aggregate_status: int = 200) -> list[str]:
    """Every rollout scores 1.0 except one, which ends in a retryable infrastructure failure."""
    urls = []

    async def post(**kwargs):
        urls.append(kwargs["url_path"])
        if kwargs["url_path"] == "/aggregate_metrics":
            if aggregate_status != 200:
                return FakeResponse(aggregate_status, {"detail": "metric hook failed"})
            rows = [dict(r) for r in kwargs["json"].verify_responses]
            return FakeResponse(200, compute_aggregate_metrics(rows).model_dump())
        body = kwargs["json"]
        if body.get("_ng_task_index") == FAILED_TASK:
            return FakeResponse(
                200,
                {"_ng_failure_class": "infrastructure_error", "failure_kind": "verifier_error", "response": {}},
            )
        return FakeResponse(200, {"reward": 1.0, "response": {}})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    return urls


async def test_a_failed_share_within_the_limit_writes_a_marked_partial_score(strict_config, monkeypatch):
    strict_config.max_failed_rollout_fraction = 0.1
    install_one_failing_rollout(monkeypatch)

    # The run is still incomplete, so a job chain retries it as before.
    with pytest.raises(IncompleteEvaluationError, match=f"{TASKS - 1}/{TASKS} samples completed"):
        await RolloutCollectionHelper().run_from_config(strict_config)

    output = Path(strict_config.output_jsonl_fpath)
    entries = orjson.loads(aggregate_metrics_path_for(output).read_bytes())
    record = {
        "accepted_failures": 1,
        "expected": TASKS,
        "accepted_failure_fraction": 1 / TASKS,
        "max_failed_rollout_fraction": 0.1,
    }
    assert [entry[PARTIAL_SCORE_KEY] for entry in entries] == [record]
    # The failed rollout is in the denominator as a zero.
    assert entries[0]["agent_metrics"]["mean/reward"] == pytest.approx((TASKS - 1) / TASKS)
    coverage = json.loads(coverage_path_for(output).read_text())
    assert coverage[PARTIAL_SCORE_KEY] == record
    assert (coverage["scored"], coverage["complete"]) == (TASKS, False)


@pytest.mark.parametrize("limit", [None, 0.01], ids=["unset", "above-limit"])
async def test_without_the_option_or_above_the_limit_no_score_is_written(strict_config, monkeypatch, limit):
    strict_config.max_failed_rollout_fraction = limit
    output = Path(strict_config.output_jsonl_fpath)
    aggregate_metrics_path_for(output).write_text("[]")  # an earlier segment's aggregate describes other rows
    urls = install_one_failing_rollout(monkeypatch)

    with pytest.raises(IncompleteEvaluationError):
        await RolloutCollectionHelper().run_from_config(strict_config)

    assert "/aggregate_metrics" not in urls
    assert not aggregate_metrics_path_for(output).exists()
    assert PARTIAL_SCORE_KEY not in json.loads(coverage_path_for(output).read_text())


async def test_a_metric_hook_that_fails_on_the_zero_rows_leaves_no_score(strict_config, monkeypatch):
    strict_config.max_failed_rollout_fraction = 0.1
    urls = install_one_failing_rollout(monkeypatch, aggregate_status=500)

    # The run's own rejection stands, not an aggregation error, so the chain decides its retry as before.
    with pytest.raises(IncompleteEvaluationError):
        await RolloutCollectionHelper().run_from_config(strict_config)

    output = Path(strict_config.output_jsonl_fpath)
    assert "/aggregate_metrics" in urls
    assert not aggregate_metrics_path_for(output).exists()
    coverage = json.loads(coverage_path_for(output).read_text())
    assert PARTIAL_SCORE_KEY not in coverage and coverage["scored"] == TASKS - 1
