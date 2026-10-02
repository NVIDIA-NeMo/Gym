# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import nemo_gym.rollout_collection as collection
from nemo_gym.rollout_collection import RolloutCollectionHelper
from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client
from tests.unit_tests.test_rollout_recovery import runner_config  # noqa: F401


@pytest.mark.parametrize("retry_failure", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("alias", [False, True])
async def test_resume_preserves_health_and_isolates_failed_attempt(
    runner_config, monkeypatch, retry_failure, workers, alias
):
    from nemo_gym.rollout_health import run_health_checks
    from tests.unit_tests.test_rollout_health import _record

    runner_config.disable_aggregation = runner_config.disable_health_check = False
    runner_config.health_check_workers = workers
    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", AsyncMock(return_value=None))
    monkeypatch.setattr(collection, "get_exporters", list)

    async def post(**kwargs):
        row = kwargs["json"]
        first = row.get("_ng_attempt_index", 0) == 0
        result = _record(row["_ng_task_index"], 0, answer="" if row["task"] == 1 and first else "ok", refs=[])
        result["reward"] = 1.0
        if retry_failure and row["task"] == 1 and first:
            result["_ng_failure_class"] = "agent_run_error"
        return FakeResponse(200, result)

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    helper = RolloutCollectionHelper()
    await helper.run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    before = output.read_bytes()
    if alias:
        shortcut = output.with_name("shortcut.jsonl")
        shortcut.symlink_to(output)
        runner_config.output_jsonl_fpath = str(shortcut)
    health_source = Path(runner_config.output_jsonl_fpath)
    original_report = json.loads((output.parent / "quality_summary.json").read_bytes())["run"]
    runner_config.resume_from_cache = True
    await helper.run_from_config(runner_config)
    assert client.post.await_count == 3 + int(retry_failure)
    assert output.read_bytes().startswith(before)
    if not retry_failure:
        assert output.read_bytes() == before
    expected = json.loads((output.parent / "quality_summary.json").read_bytes())["run"]
    assert expected["artifacts"]["records"] == 3
    assert expected["issues"]["rollout_duplicate_identity"] == 0
    assert expected["issues"]["agent_turn_hollow"] == int(not retry_failure)
    if not retry_failure:
        assert expected == original_report
    else:
        failed = run_health_checks(
            collection.failures_path_for(output), workers=1, output_dir=output.parent / "failed"
        )
        assert failed.summary["run"]["issues"]["agent_turn_hollow"] == 1
    for merge in (False, True):
        target = output.parent / f"aggregate-{merge}" / "rollouts.jsonl"
        await collection.RolloutAggregationHelper().run_from_config(
            collection.RolloutAggregationConfig(
                input_glob=str(health_source),
                output_jsonl_fpath=str(target),
                merge_shards=merge,
                health_check_workers=workers,
            )
        )
        assert json.loads((target.parent / "quality_summary.json").read_bytes())["run"] == expected
    assert (
        run_health_checks(health_source, workers=workers, output_dir=output.parent / "standalone").summary["run"]
        == expected
    )
