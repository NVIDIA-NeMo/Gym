# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from types import SimpleNamespace

import pytest

from benchmarks.agentic_if_realworld import rollout_driver
from nemo_gym.config import ATTEMPT_INDEX_KEY_NAME
from nemo_gym.rollouts import RolloutCollectionHelper
from resources_servers.agentic_if_realworld.benchmark.core import digest, judge, write
from resources_servers.agentic_if_realworld.benchmark.transport import RetryableTransportError


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
    tmp_path,
    monkeypatch,
    seen,
    *,
    parallel,
    failing=None,
    failing_write=None,
    delays=None,
    judge_override=None,
    deferring=(),
    resume=False,
    client=None,
):
    seen.update(active=0, peak=0, started=[], finished=[])

    async def run_cell(dataset, identifier, *, directory, verify, validate_only=False, **_):
        if validate_only:
            return None
        if client is not None:
            receipt = directory / "model" / (identifier + ".json")
            write(receipt, {"status": "generated"})
            selected = {
                "receipt_path": str(receipt),
                "receipt_sha256": digest(receipt),
                "rollout": {ATTEMPT_INDEX_KEY_NAME: 0, "responses_create_params": {}, "response": {}},
            }
            await verify(selected, None)
        seen["started"].append(identifier)
        seen["active"] += 1
        seen["peak"] = max(seen["peak"], seen["active"])
        await asyncio.sleep((delays or {}).get(identifier, 0.01))
        seen["active"] -= 1
        if identifier == failing:
            raise ValueError("Ambiguous upstream dispatch")
        # A deferred task keeps its journal: its answer is saved and only the judge call waits.
        write(directory / "cells" / (judge.sha(identifier) + ".json"), {"attempts": [{}]})
        if identifier in deferring:
            raise rollout_driver.DeferredJudgeError("Judge transport failed (http_status=500)")
        seen["finished"].append(identifier)
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
    monkeypatch.setattr(RolloutCollectionHelper, "setup_server_client", lambda self: client)
    monkeypatch.setattr(RolloutCollectionHelper, "preprocess_examples", lambda self, rows, **_: rows)
    config = SimpleNamespace(
        num_repeats=1,
        fan_out=None,
        responses_create_params=None,
        input_jsonl_fpath=str(tmp_path / "input.jsonl"),
        output_jsonl_fpath=str(tmp_path / "out" / "rollouts.jsonl"),
        limit=None,
        resume_from_cache=resume,
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


@pytest.mark.parametrize("resume", [False, True])
def test_deferred_transport_failure_lets_every_other_task_run_and_publish(tmp_path, monkeypatch, resume):
    # An auto-resume chain sets resume_from_cache on its first segment too, before anything is published.
    seen = {}
    with pytest.raises(RetryableTransportError, match="http_status=500"):
        collect(tmp_path, monkeypatch, seen, parallel=2, deferring={"task-0"}, resume=resume)
    assert seen["started"] == TASKS
    published = [
        json.loads(line)["task_id"] for line in (tmp_path / "out" / "rollouts.jsonl").read_text().splitlines()
    ]
    assert sorted(published) == sorted(seen["aggregated"]) == TASKS[1:]


@pytest.mark.parametrize("status", [401, 403, 503])
def test_verify_failure_admission(tmp_path, monkeypatch, status):
    from unittest.mock import AsyncMock

    from aiohttp import ClientResponseError

    from nemo_gym.cli.utils import exit_cleanly_on_config_error
    from nemo_gym.client import http_client
    from responses_api_agents.simple_agent.tests.test_app import _mock_response

    async def unavailable(response):
        error = ClientResponseError(request_info=None, history=(), status=status)
        error.response_content = b'{"detail": "Judge transport failed (http_status=500, reason=HTTP_500)"}'
        raise error

    monkeypatch.setattr(http_client, "raise_for_status", unavailable)
    post = AsyncMock(return_value=_mock_response(status=status))
    seen = {}

    marker = tmp_path / ".gym_driver_failure.100"
    monkeypatch.setenv("GYM_DRIVER_FAILURE_FILE", str(marker))
    run = exit_cleanly_on_config_error(collect)
    if status in (401, 403):
        from nemo_gym.core import NgFailureEvent
        from nemo_gym.core.failure_kinds import PROVIDER_CREDENTIALS_REJECTED

        with pytest.raises(NgFailureEvent, match="Judge credentials rejected") as caught:
            run(tmp_path, monkeypatch, seen, parallel=1, client=SimpleNamespace(post=post))
        assert caught.value.failure_kind == PROVIDER_CREDENTIALS_REJECTED
        assert marker.read_text().strip() == PROVIDER_CREDENTIALS_REJECTED
        assert post.await_count == 1
        return
    with pytest.raises(rollout_driver.DeferredJudgeError, match="503 for task-0: .*http_status=500") as caught:
        run(tmp_path, monkeypatch, seen, parallel=1, client=SimpleNamespace(post=post))
    # Every task still reached its own /verify after task-0's judge was unavailable.
    assert marker.read_text().strip() == caught.value.failure_kind == "judge_failed"
    assert post.await_count == len(TASKS)


def test_resume_judges_the_deferred_task_again_and_completes(tmp_path, monkeypatch):
    seen = {}
    with pytest.raises(RetryableTransportError):
        collect(tmp_path, monkeypatch, seen, parallel=2, deferring={"task-0"})
    with pytest.raises(RetryableTransportError):
        collect(tmp_path, monkeypatch, seen, parallel=2, deferring={"task-0"}, resume=True)
    output = tmp_path / "out" / "rollouts.jsonl"
    assert sorted(json.loads(line)["task_id"] for line in output.read_text().splitlines()) == TASKS[1:]
    collect(tmp_path, monkeypatch, seen, parallel=2, resume=True)
    assert sorted(json.loads(line)["task_id"] for line in output.read_text().splitlines()) == TASKS


def test_halted_resume_keeps_the_published_rows(tmp_path, monkeypatch):
    seen = {}
    with pytest.raises(RetryableTransportError):
        collect(tmp_path, monkeypatch, seen, parallel=1, deferring={"task-5"})
    output = tmp_path / "out" / "rollouts.jsonl"
    before = output.read_bytes()
    with pytest.raises(ValueError, match="Ambiguous upstream dispatch"):
        collect(tmp_path, monkeypatch, seen, parallel=1, failing="task-0", resume=True)
    assert seen["started"] == ["task-0"]
    assert output.read_bytes() == before


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
