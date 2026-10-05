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

"""Behavioral checks for structured outcomes and resuming an interrupted collector."""

import asyncio
import json
import multiprocessing
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

import nemo_gym.rollout_collection as collection
from nemo_gym.config_types import ConfigError
from nemo_gym.episode_types import EpisodeFailure, EpisodeId
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper, _CompletedRollout
from nemo_gym.rollout_journal import (
    RolloutJournal,
    coverage_path_for,
    journal_path_for,
    logical_rollout_id,
    read_records,
)
from nemo_gym.rollout_outcomes import RolloutFailure
from nemo_gym.rollout_recovery import RunManifest, manifest_path_for, validate_resume
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_rollout_collection import FakeResponse, failing_row, http_error, install_fake_server_client


def echo_failure(connection, failure):
    """Exercise the spawn/pickle boundary used by process-based callers."""
    connection.send(failure)
    connection.close()


def test_failure_survives_spawned_process():
    failure = RolloutFailure(
        episode_id=EpisodeId(rollout_id="task-42", attempt=0),
        run_id="test-run",
        source="environment",
        delivery="delivered",
        failure=EpisodeFailure(
            failure_reason="Timed out", terminal=False, failure_kind="transport_timeout", stage=None
        ),
    )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=echo_failure, args=(child, failure))
    process.start()
    child.close()
    try:
        assert parent.poll(30), "Child did not return its failure record"
        assert parent.recv() == failure
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"reward": 0},
        {"reward": float("nan"), "response": {}},
        {"reward": True, "response": {}},
        {"_ng_failure_class": []},
        {"type": "failure", "reward": 0},
    ],
)
async def test_malformed_results_become_associated_failures(payload, monkeypatch):
    install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, payload)))
    row = failing_row()
    original, outcome = await next(RolloutCollectionHelper().run_outcomes([row]))
    assert original == row | {"_ng_run_id": outcome.run_id}
    assert isinstance(outcome, RolloutFailure)
    assert outcome.failure.stage is None
    assert outcome.exception_type == "InvalidRolloutResult"
    assert outcome.episode_id.rollout_id
    assert "reward" not in outcome.model_dump()


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("failure_kind", [None, "verifier_error"])
async def test_valid_zero_and_masked_completed_results_remain_results(monkeypatch, masked, failure_kind):
    result = {
        "reward": 0.0,
        "response": {},
        "mask_sample": masked,
        "failure_kind": failure_kind,
        "failure_reason": "Verifier degraded" if failure_kind else None,
    }
    install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, result)))
    _, outcome = await next(RolloutCollectionHelper().run_outcomes([failing_row()]))
    assert outcome is result


@pytest.mark.parametrize("wrong_identity", [None, "rollout_id", "attempt_index", "run_id"])
async def test_agent_can_return_a_failure_for_its_dispatched_attempt(monkeypatch, wrong_identity):
    row = failing_row() | {"_ng_rollout_id": "stable-task", "_ng_attempt_index": 2, "_ng_run_id": "test-run"}
    failure = RolloutFailure(
        episode_id=EpisodeId(rollout_id="stable-task", attempt=2),
        run_id="test-run",
        source="environment",
        delivery="delivered",
        failure=EpisodeFailure(
            failure_reason="Judge unavailable", terminal=False, failure_kind="judge_failed", stage="verification"
        ),
    )
    payload = failure.model_dump()
    if wrong_identity == "rollout_id":
        payload["episode_id"]["rollout_id"] = "another-task"
    elif wrong_identity == "attempt_index":
        payload["episode_id"]["attempt"] = 1
    elif wrong_identity == "run_id":
        payload["run_id"] = "another-run"
    install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, payload)))
    original, outcome = await next(RolloutCollectionHelper().run_outcomes([row]))
    assert original == row | {"_ng_run_id": outcome.run_id}
    if wrong_identity is None:
        assert outcome == failure
    else:
        assert outcome.episode_id.rollout_id == failure.episode_id.rollout_id
        assert outcome.episode_id.attempt == failure.episode_id.attempt
        assert outcome.exception_type == "InvalidRolloutResult"
        assert outcome.failure.stage is None


@pytest.mark.parametrize("failure_class", ["judge_failed", "judge_invalid"])
async def test_legacy_judge_placeholder_is_converted_only_for_typed_callers(monkeypatch, failure_class):
    legacy = {
        "reward": 0.0,
        "response": {"output": []},
        "_ng_failure_class": failure_class,
        "failure_reason": "Judge unavailable",
    }
    install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, legacy)))
    _, raw = await next(RolloutCollectionHelper().run_examples([failing_row()]))
    assert raw is legacy
    _, outcome = await next(RolloutCollectionHelper().run_outcomes([failing_row()]))
    assert isinstance(outcome, RolloutFailure)
    assert outcome.failure.failure_kind == failure_class
    assert "response" not in outcome.model_dump()


async def test_expected_failure_does_not_stop_independent_work(monkeypatch):
    rows = [failing_row(), failing_row() | {"_ng_task_index": 99}]

    async def post(**kwargs):
        if kwargs["json"]["_ng_task_index"] == rows[0]["_ng_task_index"]:
            raise http_error(503)
        return FakeResponse(200, {"reward": 0.0, "response": {}})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    outcomes = [await future for future in RolloutCollectionHelper().run_outcomes(rows)]
    assert sum(isinstance(result, RolloutFailure) for _, result in outcomes) == 1
    assert [result["reward"] for _, result in outcomes if isinstance(result, dict)] == [0.0]


@pytest.mark.parametrize("error", [RuntimeError("programming error"), asyncio.CancelledError()])
async def test_typed_api_preserves_programming_errors_and_cancellation(error, monkeypatch):
    install_fake_server_client(monkeypatch, AsyncMock(side_effect=error))
    with pytest.raises(type(error)):
        await next(RolloutCollectionHelper().run_outcomes([failing_row()]))


def test_typed_api_requires_identity():
    with pytest.raises(ValueError, match="preprocess_examples"):
        RolloutCollectionHelper().run_outcomes([{"agent_ref": {"name": "agent"}}])


@pytest.fixture
def saved_manifest(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"question":"A"}\n')
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0, "question": "A"}]
    materialized = tmp_path / "materialized.jsonl"
    materialized.write_text(json.dumps(rows[0]) + "\n")
    configuration = {"limit": 1, "responses_create_params": {"temperature": 0.3}}
    servers = {
        "policy": {
            "responses_api_models": {
                "vllm_model": {
                    "model": "model-A",
                    "host": "host-A",
                    "port": 100,
                    "api_key": "credential",
                }
            }
        }
    }
    manifest = RunManifest.create(source, rows, configuration, servers)
    path = tmp_path / "manifest.json"
    manifest.write(path)
    return source, rows, materialized, configuration, servers, manifest, path


def test_resume_allows_new_server_addresses_and_operational_options(saved_manifest):
    source, rows, materialized, config, servers, saved, path = saved_manifest
    config = config | {
        "resume_from_cache": True,
        "num_samples_in_parallel": 10,
        "count_missing_rollouts_as_zero": True,
    }
    servers["policy"]["responses_api_models"]["vllm_model"].update(host="host-B", port=200, api_key="new")
    current = RunManifest.create(source, rows, config, servers)
    assert validate_resume(path, current, materialized).run_id == saved.run_id
    assert "credential" not in path.read_text()
    assert "model-A" not in path.read_text()


@pytest.mark.parametrize("changed", ["source", "materialized", "sampling", "model", "tool_schema"])
def test_resume_rejects_changed_identity(saved_manifest, changed):
    source, rows, materialized, config, servers, saved, path = saved_manifest
    if changed == "source":
        source.write_text('{"question":"B"}\n')
    elif changed == "materialized":
        materialized.write_text('{"question":"corrupted"}\n')
    elif changed == "sampling":
        config["responses_create_params"]["temperature"] = 0.8
    elif changed == "model":
        servers["policy"]["responses_api_models"]["vllm_model"]["model"] = "model-B"
    else:
        servers["policy"]["responses_api_models"]["vllm_model"]["tools"] = {"properties": {"port": {"type": "string"}}}
    current = RunManifest.create(source, rows, config, servers)
    with pytest.raises(ConfigError, match="incompatible"):
        validate_resume(path, current, materialized)
    assert RunManifest.model_validate_json(path.read_bytes()) == saved


def test_legacy_resume_requires_explicit_override(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(ConfigError, match="no run manifest"):
        validate_resume(path, None, tmp_path / "input.jsonl")
    with pytest.warns(UserWarning, match="allow_unsafe_resume"):
        assert validate_resume(path, None, tmp_path / "input.jsonl", allow_unsafe=True) is None


def test_unknown_manifest_version_is_rejected_even_with_override(saved_manifest):
    _, _, materialized, _, _, saved, path = saved_manifest
    path.write_text(json.dumps(saved.model_dump() | {"schema_version": 99}))
    with pytest.raises(ConfigError, match="Cannot read"):
        validate_resume(path, None, materialized, allow_unsafe=True)


def test_older_manifest_defaults_to_existing_attempt_selection_policy(saved_manifest):
    _, _, _, _, _, saved, _ = saved_manifest
    payload = saved.model_dump()
    del payload["selection_policy"]
    assert RunManifest.model_validate(payload).selection_policy == "latest_dispatched"


def test_unknown_selection_policy_is_rejected_even_with_override(saved_manifest):
    _, _, materialized, _, _, saved, path = saved_manifest
    path.write_text(json.dumps(saved.model_dump() | {"selection_policy": "any_success"}))
    with pytest.raises(ConfigError, match="Cannot read"):
        validate_resume(path, None, materialized, allow_unsafe=True)


@pytest.mark.parametrize("interruption", [asyncio.CancelledError, RuntimeError])
async def test_interrupted_runner_closes_files_and_reuses_saved_zero(tmp_path, monkeypatch, interruption):
    client = install_fake_server_client(monkeypatch, AsyncMock())
    client.global_config_dict = OmegaConf.create({"agent": {"responses_api_agents": {"impl": {}}}})
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})
    source = tmp_path / "input.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "responses_create_params": {"input": []},
                    "agent_ref": {"name": "agent"},
                    "task": task,
                }
            )
            + "\n"
            for task in range(3)
        )
    )
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        disable_aggregation=True,
        disable_health_check=True,
    )
    opened = []
    original_open = Path.open

    def track_open(path, *args, **kwargs):
        file = original_open(path, *args, **kwargs)
        if args and args[0] == "ab":
            opened.append(file)
        return file

    monkeypatch.setattr(Path, "open", track_open)
    dispatched = []
    interrupt = True

    class Helper(RolloutCollectionHelper):
        def _run_examples_with_metadata(self, examples, **kwargs):
            # Yield lazily to interrupt deterministically after the first flushed result.
            for row in examples:

                async def complete(row=row):
                    kwargs["on_dispatch"](row)
                    dispatched.append(row["task"])
                    if interrupt and row["task"] == 1:
                        raise interruption()
                    return _CompletedRollout(row=row, result={"reward": 0.0, "response": {}}, rollout_latency_ms=None)

                yield complete()

    helper = Helper()
    with pytest.raises(interruption):
        await helper.run_from_config(config)
    assert opened and all(file.closed for file in opened)
    before = Path(config.output_jsonl_fpath).read_bytes()
    assert len(before.splitlines()) == 1
    run_id = RunManifest.model_validate_json(manifest_path_for(Path(config.output_jsonl_fpath)).read_bytes()).run_id
    interrupt = False
    dispatched.clear()
    config.resume_from_cache = True
    await helper.run_from_config(config)
    assert dispatched == [1, 2]
    assert all(file.closed for file in opened)
    after = Path(config.output_jsonl_fpath).read_bytes()
    assert after.startswith(before)
    assert len(after.splitlines()) == 3
    resumed = [json.loads(line) for line in after.splitlines()]
    assert resumed[1]["_ng_attempt_index"] == 1
    assert resumed[2].get("_ng_attempt_index", 0) == 0
    assert (
        RunManifest.model_validate_json(manifest_path_for(Path(config.output_jsonl_fpath)).read_bytes()).run_id
        == run_id
    )
    dispatched.clear()
    await helper.run_from_config(config)
    assert dispatched == []


async def test_runner_rejects_changed_inputs_before_dispatch_or_output_mutation(tmp_path, monkeypatch):
    client = install_fake_server_client(monkeypatch, AsyncMock())
    client.global_config_dict = OmegaConf.create({"agent": {"responses_api_agents": {"impl": {}}}})
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "agent"}}) + "\n")
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        disable_aggregation=True,
        disable_health_check=True,
    )
    calls = []

    class Helper(RolloutCollectionHelper):
        def _run_examples_with_metadata(self, examples, **kwargs):
            for row in examples:

                async def complete(row=row):
                    calls.append(row)
                    return _CompletedRollout(row=row, result={"reward": 0.0, "response": {}}, rollout_latency_ms=None)

                yield complete()

    helper = Helper()
    await helper.run_from_config(config)
    output = Path(config.output_jsonl_fpath)
    original = output.read_bytes()
    source.write_text(source.read_text().replace('"input": []', '"input": "changed"'))
    config.resume_from_cache = True
    with pytest.raises(ConfigError, match="incompatible"):
        await helper.run_from_config(config)
    assert len(calls) == 1
    assert output.read_bytes() == original


def test_identity_override_remains_visible_on_future_resumes(saved_manifest):
    source, rows, materialized, config, servers, saved, path = saved_manifest
    changed = RunManifest.create(source, rows, config | {"limit": 2}, servers)
    with pytest.warns(UserWarning, match="allow_unsafe_resume"):
        overridden = validate_resume(path, changed, materialized, allow_unsafe=True)
    assert overridden.identity_overridden and overridden.run_id == saved.run_id
    overridden.write(path)
    with pytest.raises(ConfigError, match="previously overridden"):
        validate_resume(path, saved, materialized)


@pytest.fixture
def runner_config(tmp_path, monkeypatch):
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(failing_row(task) | {"task": task}) + "\n" for task in range(3)))
    return RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        route_failures_to_sidecar=True,
        disable_aggregation=True,
        disable_health_check=True,
        num_samples_in_parallel=1,
    )


@pytest.mark.parametrize("retain_results", [False, True])
@pytest.mark.parametrize("masked_unscored", [False, True])
async def test_native_recovery_preserves_terminal_failures_and_unscored_completion(
    tmp_path, monkeypatch, retain_results, masked_unscored
):
    from nemo_gym.rollout_store import RolloutStore

    source = tmp_path / "tasks.jsonl"
    tasks = [{"task_id": {"taskset": "native", "task_id": str(i)}, "task_input": {"scenario": i}} for i in range(3)]
    source.write_text("".join(json.dumps(row) + "\n" for row in tasks))
    calls = []

    async def post(**kwargs):
        assert kwargs["url_path"] == "/run" and kwargs["server_name"] == "environment"
        request = kwargs["json"]
        task = request["task"]["task_id"]["task_id"]
        attempt = request["episode_id"]["attempt"]
        calls.append((task, attempt))
        identity = {"episode_id": request["episode_id"], "task_id": request["task"]["task_id"]}
        if task == "0":
            # A custom protocol can complete without a score or an LLM response.
            return FakeResponse(200, identity | {"result": {"artifact": "done", "mask_sample": masked_unscored}})
        if task == "1" and attempt == 1:
            return FakeResponse(200, identity | {"result": {"reward": 0.0}})
        return FakeResponse(200, identity | {"failure": {"failure_reason": "setup failed", "terminal": task == "2"}})

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    client.global_config_dict = OmegaConf.create({"environment": {"environment_servers": {"custom": {"scenario": 1}}}})
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        environment_server_routes={"native": "environment"},
        route_failures_to_sidecar=True,
        retain_results_in_memory=retain_results,
        max_resident_rollout_tasks=1,
        disable_aggregation=True,
        disable_health_check=True,
    )
    await RolloutCollectionHelper().run_from_config(config)
    store = RolloutStore.read(Path(config.output_jsonl_fpath))
    assert store.coverage()["successful"] == 1
    assert store.coverage()["unscored"] == 1 and store.coverage()["measured"] == 0
    assert store.selected("success")[0]["artifact"] == "done"
    assert all("reward" not in failure for failure in store.failures())
    assert all(
        failure["_ng_failure_record"]["failure"]["failure_reason"] == "setup failed" for failure in store.failures()
    )
    assert all(failure["_ng_failure_record"]["failure"].get("stage") is None for failure in store.failures())
    config.resume_from_cache = True
    await RolloutCollectionHelper().run_from_config(config)
    store = RolloutStore.read(Path(config.output_jsonl_fpath))
    assert calls == [("0", 0), ("1", 0), ("2", 0), ("1", 1)]
    assert store.coverage()["successful"] == 2
    assert store.coverage()["measured"] == 1 and store.coverage()["unscored"] == 1
    assert store.coverage()["failed"] == 1
    assert store.selected("success")[1]["reward"] == 0.0
    config.require_complete = True
    with pytest.raises(RuntimeError, match="2/3 samples completed"):
        await RolloutCollectionHelper().run_from_config(config)
    assert len(calls) == 4  # The unscored completion and terminal failure are not retried.
    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", AsyncMock(return_value=None))
    merged = Path(config.output_jsonl_fpath).with_name("merged.jsonl")
    await collection.RolloutAggregationHelper().run_from_config(
        collection.RolloutAggregationConfig(
            input_glob=config.output_jsonl_fpath,
            output_jsonl_fpath=str(merged),
            disable_health_check=True,
        )
    )
    offline = json.loads(coverage_path_for(merged).read_text())
    assert (offline["successful"], offline["measured"], offline["masked"], offline["unscored"], offline["scored"]) == (
        2,
        1,
        0,
        1,
        1,
    )


@pytest.mark.parametrize("mismatch", ["attempt", "task", "reserved_result"])
async def test_native_typed_outcomes_reject_foreign_identity_without_stopping_other_work(monkeypatch, mismatch):
    rows = [
        {
            "task_id": {"taskset": "native", "task_id": str(i)},
            "task_input": {},
            "_ng_task_index": i,
            "_ng_rollout_index": 0,
            "_ng_environment_server": "environment",
        }
        for i in range(2)
    ]

    async def post(**kwargs):
        request = kwargs["json"]
        result = {
            "episode_id": dict(request["episode_id"]),
            "task_id": dict(request["task"]["task_id"]),
            "result": {"artifact": "completed"},
        }
        if result["task_id"]["task_id"] == "0":
            if mismatch == "attempt":
                result["episode_id"]["attempt"] = 5
            elif mismatch == "task":
                result["task_id"]["task_id"] = "foreign"
            else:
                result["result"]["_ng_run_id"] = "foreign"
        return FakeResponse(200, result)

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    client.global_config_dict = OmegaConf.create({"environment": {"environment_servers": {"custom": {}}}})
    outcomes = {
        row["_ng_task_index"]: result
        for row, result in await asyncio.gather(*RolloutCollectionHelper().run_outcomes(rows))
    }
    assert isinstance(outcomes[0], RolloutFailure) and outcomes[0].failure.stage is None
    assert outcomes[0].episode_id.rollout_id == "0-0" and outcomes[0].episode_id.attempt == 0
    assert outcomes[1]["artifact"] == "completed"


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reference", ["literal", "name", "mapping"])
def test_resume_identity_includes_environment_protocol_but_ignores_its_runtime_address(
    saved_manifest, native, reference
):
    source, rows, _, config, _, _, _ = saved_manifest
    servers = {
        "agent_name": "agent",
        "agent_reference": {"name": "${agent_name}"},
        "agent": {"responses_api_agents": {"impl": {}}},
        "environment": {
            "environment_servers": {
                "single_agent_turn": {
                    "agent_server": {"name": "agent"},
                    "resources_server": {"name": "judge"},
                    "scenario": 1,
                    "port": 8000,
                }
            }
        },
        "judge": {"resources_servers": {"impl": {"scoring_rule": 1}}},
        "unused": {"environment_servers": {"custom": {"scenario": "${oc.env:GYM_UNUSED_TEST_ENVIRONMENT}"}}},
    }
    settings = servers["environment"]["environment_servers"]["single_agent_turn"]
    if reference == "name":
        settings["agent_server"] = {"name": "${agent_name}"}
    elif reference == "mapping":
        settings["agent_server"] = "${agent_reference}"
    rows = [dict(rows[0], agent_ref={"name": "agent"})]
    if native:
        rows[0]["_ng_environment_server"] = "environment"
        rows[0].pop("agent_ref")
    before = RunManifest.create(source, rows, config, servers).config_digest
    settings["port"] = 9000
    assert (
        RunManifest.create(
            source, rows, config | {"retain_results_in_memory": False, "max_resident_rollout_tasks": 1}, servers
        ).config_digest
        == before
    )
    settings["scenario"] = 2
    assert RunManifest.create(source, rows, config, servers).config_digest != before
    settings["scenario"] = 1
    servers["judge"]["resources_servers"]["impl"]["scoring_rule"] = 2
    assert RunManifest.create(source, rows, config, servers).config_digest != before


@pytest.mark.parametrize("route_failures", [False, True])
@pytest.mark.parametrize("reason_key", ["error_message", "agent_error", "grading_notes"])
async def test_failure_sidecar_preserves_producer_diagnostics(runner_config, monkeypatch, route_failures, reason_key):
    runner_config.route_failures_to_sidecar = route_failures
    reason = "The sandbox disconnected before producing a deliverable"
    diagnostics = {
        reason_key: reason,
        "raw_rollout": {"archived_to": "saved-trace.json"},
        "hermes_result_path": "saved-hermes-trace.json",
        "verifier_reward": 0.25,
        "hermes_return_code": 1,
    }

    async def post(**kwargs):
        if kwargs["json"]["_ng_task_index"] == 1:
            return FakeResponse(
                200,
                diagnostics | {"_ng_failure_class": "agent_run_error", "reward": 0.0, "response": {"output": []}},
            )
        return FakeResponse(200, {"reward": 0.0, "response": {}})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    returned = await RolloutCollectionHelper().run_from_config(runner_config)
    [saved] = read_records(collection.failures_path_for(Path(runner_config.output_jsonl_fpath)))
    for key, value in diagnostics.items():
        assert saved[key] == value
    assert saved["_ng_failure_record"]["failure"]["failure_reason"] == reason
    assert "reward" not in saved and "response" not in saved
    assert next(row for row in returned if row["_ng_task_index"] == 1) == saved


async def test_unbounded_interruption_consumes_attempts_until_the_cap_is_raised(tmp_path, monkeypatch):
    from nemo_gym.rollout_store import RolloutStore

    rows = [failing_row(index) for index in range(4)]
    source = tmp_path / "inputs.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "rollouts.jsonl"

    def prepare():
        return rows, RunManifest.create(source, rows, {}, {"my_agent": {"responses_api_agents": {"impl": {}}}})

    for restart in range(3):
        queued = []
        sent = []
        all_queued = asyncio.Event()
        blocked = asyncio.Event()
        pool = asyncio.Semaphore(1)

        async def post(**kwargs):
            queued.append(kwargs["json"]["_ng_task_index"])
            if len(queued) == len(rows):
                all_queued.set()
            # Simulate requests waiting in the HTTP client after Gym's dispatch.
            async with pool:
                sent.append(kwargs["json"]["_ng_task_index"])
                await blocked.wait()

        install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        with RolloutStore.start_or_resume(output, prepare, resume=restart > 0) as store:
            pending = store.pending(3)
            assert len(pending) == len(rows)
            dispatches = RolloutCollectionHelper()._run_examples_with_metadata(
                pending, on_dispatch=store.record_dispatch
            )
            waiter = asyncio.create_task(next(dispatches))
            try:
                await asyncio.wait_for(all_queued.wait(), 5)
                assert len(sent) == 1
            finally:
                waiter.cancel()
                await dispatches.aclose()
                await asyncio.gather(waiter, return_exceptions=True)

    reader = RolloutStore.read(output)
    assert reader.coverage()["unknown"] == len(rows)
    assert reader.pending(3) == []
    retry = reader.pending(4)
    assert len(retry) == len(rows)
    assert all(row["_ng_attempt_index"] == 3 for row in retry)
    assert reader.selected("success") == [] and reader.failures() == []


async def test_runner_accepts_nested_hydra_overrides_and_unused_unresolved_server(runner_config, monkeypatch):
    global_config = {
        "unused": {"responses_api_models": {"openai_model": {"model": "${oc.env:NG_MISSING_REVIEW_TEST}"}}}
    }
    monkeypatch.delenv("NG_MISSING_REVIEW_TEST", raising=False)
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: global_config)
    runner_config.responses_create_params = {"metadata": OmegaConf.create({"nested": {"values": [1, 2]}})}
    client = install_fake_server_client(
        monkeypatch, AsyncMock(return_value=FakeResponse(200, {"reward": 0, "response": {}}))
    )
    await RolloutCollectionHelper().run_from_config(runner_config)
    assert client.post.await_count == 3
    for call in client.post.await_args_list:
        assert call.kwargs["json"]["responses_create_params"]["metadata"] == {"nested": {"values": [1, 2]}}
    runner_config.resume_from_cache = True
    await RolloutCollectionHelper().run_from_config(runner_config)
    assert client.post.await_count == 3


def test_identity_resolves_reachable_servers_but_ignores_operational_fields(saved_manifest, monkeypatch):
    source, rows, _, config, _, _, _ = saved_manifest
    rows = [rows[0] | {"agent_ref": {"name": "agent"}}]
    servers = {
        "policy_api_key": "old-secret",
        "policy_base_url": "http://old",
        "policy_name": "model-A",
        "agent": {
            "responses_api_agents": {
                "simple_agent": {
                    "model_server": {"name": "policy"},
                    "resources_server": {"name": "resources"},
                }
            }
        },
        "policy": {
            "responses_api_models": {
                "openai_model": {
                    "model": "${policy_name}",
                    "openai_api_key": "${policy_api_key}",
                    "openai_base_url": "${policy_base_url}",
                    "model_call_capture_dir": "/old/captures",
                }
            }
        },
        "resources": {
            "resources_servers": {"example": {"dataset_path": "/tasks/a", "prompt": "${oc.env:NG_REVIEW_PROMPT}"}}
        },
        "unused": {"responses_api_models": {"openai_model": {"model": "${oc.env:NG_MISSING_REVIEW_TEST}"}}},
    }
    monkeypatch.setenv("NG_REVIEW_PROMPT", "prompt-A")
    monkeypatch.delenv("NG_MISSING_REVIEW_TEST", raising=False)
    before = RunManifest.create(source, rows, config, servers).config_digest
    servers.update(policy_api_key="new-secret", policy_base_url="http://new", model_call_capture_dir="/new/captures")
    servers["policy"]["responses_api_models"]["openai_model"]["model_call_capture_dir"] = "/another/capture"
    assert RunManifest.create(source, rows, config, servers).config_digest == before
    monkeypatch.setenv("NG_REVIEW_PROMPT", "prompt-B")
    assert RunManifest.create(source, rows, config, servers).config_digest != before
    monkeypatch.setenv("NG_REVIEW_PROMPT", "prompt-A")
    servers["resources"]["resources_servers"]["example"]["dataset_path"] = "/tasks/b"
    assert RunManifest.create(source, rows, config, servers).config_digest != before


def test_capture_directory_changes_preserve_identity_but_behavior_changes_do_not(saved_manifest):
    source, rows, _, config, servers, _, _ = saved_manifest
    servers["token_id_capture"] = {"dir": "/old", "rebuild_response": True}
    model = servers["policy"]["responses_api_models"]["vllm_model"]
    model["token_id_capture"] = {"dir": "/old", "rebuild_response": True}
    before = RunManifest.create(source, rows, config, servers).config_digest
    for settings in (servers["token_id_capture"], model["token_id_capture"]):
        settings["dir"] = "/new"
    assert RunManifest.create(source, rows, config, servers).config_digest == before
    servers["token_id_capture"]["rebuild_response"] = False
    assert RunManifest.create(source, rows, config, servers).config_digest != before


@pytest.mark.parametrize(
    "repeats,fan_out,expected",
    [(1, None, 1), (3, None, 3), (1, {"my_agent": ["a", "b"]}, 2), (2, {"my_agent": ["a", "b"]}, 4)],
)
def test_explicit_ids_expand_deterministically(runner_config, repeats, fan_out, expected):
    runner_config.num_repeats = repeats
    runner_config.fan_out = fan_out
    example = failing_row() | {"_ng_rollout_id": "explicit-task"}
    helper = RolloutCollectionHelper()
    rows = helper.preprocess_examples([example], num_repeats=repeats, fan_out=fan_out)
    assert len(rows) == expected
    assert len({logical_rollout_id(row) for row in rows}) == expected
    assert helper.preprocess_examples([example], num_repeats=repeats, fan_out=fan_out) == rows
    Path(runner_config.input_jsonl_fpath).write_text(json.dumps(example) + "\n")
    assert helper._preprocess_rows_from_config(runner_config) == rows
    assert example["_ng_rollout_id"] == "explicit-task"
    if expected == 1:
        assert rows[0]["_ng_rollout_id"] == "explicit-task"


@pytest.mark.parametrize("identity", [[], "../bad", ""])
def test_invalid_explicit_identity_is_a_configuration_error(identity):
    with pytest.raises(ConfigError, match="Invalid rollout identity"):
        logical_rollout_id(failing_row() | {"_ng_rollout_id": identity})


@pytest.mark.parametrize("route_failures", [False, True])
async def test_failure_sidecar_retains_observations_and_captured_model_calls(
    runner_config, monkeypatch, route_failures
):
    from nemo_gym.base_responses_api_model import CaptureStore

    runner_config.route_failures_to_sidecar = route_failures
    capture_dir = Path(runner_config.output_jsonl_fpath).parent / "captures"
    captures = CaptureStore(capture_dir)
    monkeypatch.setattr(
        collection,
        "get_global_config_dict",
        lambda: {
            "observability_enabled": True,
            "model_call_capture_dir": str(capture_dir),
        },
    )
    observations = {"source": "test", "records": [{"kind": "agent_invocation", "invocation_id": "root"}]}

    async def post(**kwargs):
        row = kwargs["json"]
        captures.record(
            logical_rollout_id(row),
            {
                "model_call_id": "call-1",
                "dialect": "responses",
                "request": {"input": []},
                "response": {"id": "resp-1"},
            },
        )
        return FakeResponse(
            200,
            {
                "_ng_failure_class": "agent_run_error",
                "reward": 0,
                "response": {},
                "ng_agent_observations": observations,
                "_ng_failure_judge_error": "diagnostic detail",
            },
        )

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    with pytest.raises(RuntimeError, match="None of the 3 dispatched"):
        await RolloutCollectionHelper().run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    assert list(read_records(output)) == []
    failures = list(read_records(collection.failures_path_for(output)))
    assert len(failures) == 3
    for row in failures:
        assert row["ng_agent_observations"]["source"] == observations["source"]
        assert row["ng_agent_observations"]["records"][0]["invocation_id"] == "root"
        assert row["_ng_failure_judge_error"] == "diagnostic detail"
        assert row["ng_trajectory"]["invocations"][0]["invocation_id"] == "root"
        assert row["ng_trajectory"]["model_calls"][0]["response"] == {"id": "resp-1"}
        assert "reward" not in row and "response" not in row
        assert "ng_trajectory" not in row["_ng_failure_record"]


async def test_terminal_skips_can_be_explicitly_scored_as_zero(runner_config, monkeypatch):
    runner_config.disable_aggregation = False
    runner_config.count_failure_classes_as_zero = ["skipped"]
    aggregate = AsyncMock(return_value=None)
    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", aggregate)

    async def post(**kwargs):
        result = (
            {"reward": 0, "response": {}}
            if kwargs["json"]["task"] == 0
            else {
                "_ng_failure_class": "skipped",
                "_ng_failure_terminal": True,
            }
        )
        return FakeResponse(200, result)

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    assert len(aggregate.await_args.args[0]) == 3
    assert all(row["reward"] == 0 for row in aggregate.await_args.args[0])
    output = Path(runner_config.output_jsonl_fpath)
    assert len(list(read_records(output))) == 1
    report = json.loads(coverage_path_for(output).read_text())
    assert (report["intentionally_omitted"], report["scored"], report["failures_counted_as_zero"]) == (2, 3, 2)
    runner_config.resume_from_cache = True
    await RolloutCollectionHelper().run_from_config(runner_config)
    assert client.post.await_count == 3
    await collection.RolloutAggregationHelper().run_from_config(
        collection.RolloutAggregationConfig(
            input_glob=str(output),
            output_jsonl_fpath=str(output.with_name("merged.jsonl")),
            count_failure_classes_as_zero=["skipped"],
            disable_health_check=True,
        )
    )
    assert len(aggregate.await_args.args[0]) == 3


async def test_reported_kill_shaped_failures_consume_bounded_attempts(runner_config, monkeypatch, capsys):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "2")

    async def post(**kwargs):
        result = (
            {"reward": 0, "response": {}}
            if kwargs["json"]["task"] == 0
            else {
                "_ng_failure_class": "kill_shaped",
                "_ng_no_persist": True,
                "reward": 0,
                "response": {},
            }
        )
        return FakeResponse(200, result)

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    runner_config.resume_from_cache = True
    await RolloutCollectionHelper().run_from_config(runner_config)
    await RolloutCollectionHelper().run_from_config(runner_config)
    assert client.post.await_count == 5
    output = Path(runner_config.output_jsonl_fpath)
    failures = list(read_records(collection.failures_path_for(output)))
    assert len(failures) == 4
    assert all("reward" not in row and "_ng_no_persist" not in row for row in failures)
    coverage = json.loads(coverage_path_for(output).read_text())
    assert coverage["attempts"] == 5
    assert coverage["attempts_exhausted"] == 2
    assert coverage["max_rollout_attempts"] == 2
    printed = capsys.readouterr().out
    assert "attempt 2 of 2" in printed
    assert "Retry budget exhausted for 2 rollout(s) at the cap of 2" in printed
    assert str(collection.failures_path_for(output)) in printed


@pytest.mark.parametrize("count_failures_as_zero", [False, True])
async def test_runner_journals_before_request_and_resumes_only_failed_work(
    runner_config, monkeypatch, count_failures_as_zero
):
    output = Path(runner_config.output_jsonl_fpath)
    calls = []
    exported = []
    runner_config.disable_aggregation = False
    runner_config.upload_rollouts = False
    runner_config.count_failure_classes_as_zero = ["agent_request_failed"] if count_failures_as_zero else []
    monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", AsyncMock(return_value=None))
    monkeypatch.setattr(collection, "get_exporters", lambda: True)
    monkeypatch.setattr(collection, "export_metrics", lambda metrics, **kwargs: exported.append(metrics))

    async def post(**kwargs):
        row = kwargs["json"]
        calls.append(row)
        history = RolloutJournal.load(output, RunManifest.model_validate_json(manifest_path_for(output).read_bytes()))
        identity = logical_rollout_id(row)
        assert history.latest[identity] == row.get("_ng_attempt_index", 0)
        assert history.disposition(identity) == "unknown"
        if row["task"] == 1 and row.get("_ng_attempt_index", 0) == 0:
            raise http_error(503)
        if row["task"] == 2:
            return FakeResponse(
                200,
                {"_ng_failure_class": "skipped", "_ng_failure_terminal": True, "reward": 0, "response": {}},
            )
        return FakeResponse(
            200,
            {"reward": 0.0, "response": {}, "mask_sample": row["task"] == 0, "failure_kind": "verifier_error"},
        )

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    helper = RolloutCollectionHelper()
    await helper.run_from_config(runner_config)
    report = json.loads(coverage_path_for(output).read_text())
    assert [report[key] for key in ("successful", "failed", "intentionally_omitted", "unknown")] == [1, 1, 1, 0]
    assert (report["measured"], report["masked"]) == (0, 1)
    assert report["scored"] == 1 + int(count_failures_as_zero)
    assert report["failures_counted_as_zero"] == int(count_failures_as_zero)
    assert exported[-1] == {
        "coverage/expected": 3,
        "coverage/scored": report["scored"],
        "coverage/missing": 3 - report["scored"],
        "coverage/measured": 0,
        "coverage/masked": 1,
        "coverage/unscored": 0,
        "coverage/failed": 1,
        "coverage/omitted": 1,
        "coverage/unknown": 0,
        "coverage/attempts_exhausted": 0,
    }
    progress = [metrics for metrics in exported if "progress/total/rollouts_per_min" in metrics][-1]
    assert progress["progress/my_agent/masked_pct"] == 100.0
    assert progress["progress/my_agent/failed"] == 1
    assert progress["progress/my_agent/omitted"] == 1
    assert "progress/my_agent/reward_unmasked" not in progress
    assert not report["complete"] and report["reconciled"]
    failures = list(read_records(collection.failures_path_for(output)))
    assert len(failures) == 2
    assert all("reward" not in row and "response" not in row for row in failures)
    assert all(row["_ng_failure_record"]["run_id"] == report["run_id"] for row in failures)
    original_history = journal_path_for(output).read_bytes()
    runner_config.resume_from_cache = True
    calls.clear()
    await helper.run_from_config(runner_config)
    assert [(row["task"], row["_ng_attempt_index"]) for row in calls] == [(1, 1)]
    assert journal_path_for(output).read_bytes().startswith(original_history)
    report = json.loads(coverage_path_for(output).read_text())
    assert [report[key] for key in ("successful", "failed", "intentionally_omitted", "unknown")] == [2, 0, 1, 0]
    assert (report["measured"], report["masked"], report["scored"], report["failures_counted_as_zero"]) == (1, 1, 2, 0)
    assert sum(report[key] for key in ("measured", "masked", "failed", "intentionally_omitted", "unknown")) == 3
    assert exported[-1]["coverage/measured"] == 1 and exported[-1]["coverage/masked"] == 1
    assert exported[-1]["coverage/failed"] == 0
    assert len(list(read_records(collection.failures_path_for(output)))) == 2


@pytest.mark.parametrize("failure_type", ["typed", "legacy", "kill_shaped", "skipped", "suppressed"])
async def test_progress_masking_matches_persisted_outcomes(runner_config, monkeypatch, failure_type):
    source = Path(runner_config.input_jsonl_fpath)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[2]["agent_ref"]["name"] = "dropped_agent"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    runner_config.upload_rollouts = False
    exported = []
    monkeypatch.setattr(collection, "get_exporters", lambda: True)
    monkeypatch.setattr(collection, "export_metrics", lambda metrics, **kwargs: exported.append(metrics))

    async def post(**kwargs):
        row = kwargs["json"]
        if row["task"] < 2:
            return FakeResponse(200, {"reward": 0.5, "response": {}, "mask_sample": row["task"] == 1})
        if failure_type == "typed":
            result = RolloutFailure(
                episode_id=EpisodeId(rollout_id=logical_rollout_id(row), attempt=0),
                run_id=row["_ng_run_id"],
                source="environment",
                delivery="delivered",
                failure=EpisodeFailure(
                    failure_reason="Agent unavailable",
                    terminal=False,
                    failure_kind="agent_request_failed",
                    stage="agent",
                ),
            ).model_dump()
        else:
            result = {"reward": 0.0, "response": {}}
            if failure_type == "legacy":
                result["_ng_failure_class"] = "agent_request_failed"
            elif failure_type == "suppressed":
                result["_ng_no_persist"] = True
            elif failure_type == "skipped":
                result.update(_ng_failure_class="skipped", _ng_failure_terminal=True)
            else:
                result.update(_ng_failure_class="kill_shaped", _ng_no_persist=True)
        return FakeResponse(200, result)

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    client.global_config_dict["dropped_agent"] = {"responses_api_agents": {"impl": {}}}
    client.global_config_dict["dropped_environment"] = {
        "environment_servers": {
            "legacy_agent": {"agent_server": {"type": "responses_api_agents", "name": "dropped_agent"}}
        }
    }
    await RolloutCollectionHelper().run_from_config(runner_config)
    progress = [metrics for metrics in exported if "progress/total/rollouts_per_min" in metrics][-1]
    assert progress["progress/my_agent/masked_pct"] == 50.0
    assert progress["progress/my_agent/reward_unmasked"] == 50.0
    omitted = failure_type in {"skipped", "suppressed"}
    assert progress[f"progress/dropped_agent/{'omitted' if omitted else 'failed'}"] == 1
    assert not any(key.startswith("progress/dropped_agent/reward") for key in progress)
    assert exported[-1]["coverage/failed"] == int(not omitted)
    assert exported[-1]["coverage/omitted"] == int(omitted)
    assert (exported[-1]["coverage/measured"], exported[-1]["coverage/masked"]) == (1, 1)


async def test_runner_cancellation_closes_requests_before_return(runner_config, monkeypatch):
    started = asyncio.Event()
    stopped = asyncio.Event()
    requests = []

    async def post(**kwargs):
        requests.append(kwargs["json"])
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    task = asyncio.create_task(RolloutCollectionHelper().run_from_config(runner_config))
    await asyncio.wait_for(started.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set() and len(requests) == 1
    output = Path(runner_config.output_jsonl_fpath)
    report = json.loads(coverage_path_for(output).read_text())
    assert (report["attempts"], report["unknown"], report["never_dispatched"]) == (1, 3, 2)
    assert output.read_bytes() == collection.failures_path_for(output).read_bytes() == b""


@pytest.mark.parametrize("route_failures", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
async def test_runner_accepts_explicit_failures_independently_of_exception_policy(
    runner_config, monkeypatch, route_failures, legacy
):
    runner_config.route_failures_to_sidecar = route_failures

    async def post(**kwargs):
        row = kwargs["json"]
        if row["task"] == 0:
            return FakeResponse(200, {"reward": 0, "response": {}})
        if legacy:
            return FakeResponse(200, {"_ng_failure_class": "judge_failed", "error": "Judge unavailable"})
        return FakeResponse(
            200,
            RolloutFailure(
                episode_id=EpisodeId(rollout_id=logical_rollout_id(row), attempt=0),
                run_id=row["_ng_run_id"],
                source="environment",
                delivery="delivered",
                failure=EpisodeFailure(
                    failure_reason="Judge unavailable",
                    terminal=False,
                    failure_kind="judge_failed",
                    stage="verification",
                ),
            ).model_dump(),
        )

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    assert len(list(read_records(output))) == 1
    failures = list(read_records(collection.failures_path_for(output)))
    assert len(failures) == 2 and all("reward" not in row for row in failures)
    for row in failures:
        assert row["error"] == "Judge unavailable"
        assert row["_ng_failure_message"] == row["_ng_failure_record"]["failure"]["failure_reason"] == row["error"]


@pytest.mark.parametrize(
    "field",
    [
        "judge_api_key",
        "judge_base_url",
        "tavily_api_key",
        "anthropic_api_key",
        "sandbox_model_base_url",
        "num_workers",
        "switchyard_api_key",
        "switchyard_base_url",
    ],
)
def test_known_operational_settings_do_not_change_resume_identity(saved_manifest, field):
    source, rows, _, config, servers, _, _ = saved_manifest
    settings = servers["policy"]["responses_api_models"]["vllm_model"]
    settings[field] = "before"
    before = RunManifest.create(source, rows, config, servers).config_digest
    settings[field] = "after"
    assert RunManifest.create(source, rows, config, servers).config_digest == before
    # The same spelling inside task data must not be silently erased.
    settings["task_parameters"] = {field: "before"}
    before = RunManifest.create(source, rows, config, servers).config_digest
    settings["task_parameters"][field] = "after"
    assert RunManifest.create(source, rows, config, servers).config_digest != before


@pytest.mark.parametrize(
    "headers", ["headers", "default_headers", "openai_default_headers", "artifact_request_headers"]
)
def test_auth_headers_need_not_resolve_but_task_headers_still_affect_identity(saved_manifest, monkeypatch, headers):
    source, rows, _, config, servers, _, _ = saved_manifest
    monkeypatch.delenv("GYM_TEST_MISSING_CREDENTIAL", raising=False)
    settings = servers["policy"]["responses_api_models"]["vllm_model"]
    settings[headers] = {"Authorization": "Bearer ${oc.env:GYM_TEST_MISSING_CREDENTIAL}", "X-Dataset-Version": "v1"}
    before = RunManifest.create(source, rows, config, servers).config_digest
    settings[headers]["Authorization"] = "Bearer changed"
    assert RunManifest.create(source, rows, config, servers).config_digest == before
    settings[headers]["X-Dataset-Version"] = "v2"
    assert RunManifest.create(source, rows, config, servers).config_digest != before


@pytest.mark.parametrize("native_tokens", [False, True])
async def test_judge_failure_saves_token_evidence_without_retiring_capture(
    runner_config, monkeypatch, tmp_path, native_tokens
):
    from nemo_gym.token_id_capture import TokenCaptureStore
    from tests.unit_tests.test_rollout_collection import TestFinalizeRolloutTokenCapture

    captures = TokenCaptureStore(tmp_path / "tokens")
    settings = {"token_id_capture": {"enabled": True, "all_agents": True, "dir": str(tmp_path / "tokens")}}
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: settings)
    monkeypatch.setattr(collection, "installed_token_source", lambda: captures)
    response = {"model": "m", "output": []}
    if native_tokens:
        response["output"] = [{"type": "message", "role": "assistant", "content": [], "generation_token_ids": [77]}]

    async def post(**kwargs):
        if kwargs["json"]["task"] == 0:
            TestFinalizeRolloutTokenCapture._capture(captures)
            return FakeResponse(200, {"_ng_failure_class": "judge_failed", "reward": 0.0, "response": response})
        return FakeResponse(200, {"_ng_failure_class": "agent_run_error"})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    with pytest.raises(RuntimeError, match="None of the 3 dispatched"):
        await RolloutCollectionHelper().run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    failures = list(read_records(collection.failures_path_for(output)))
    judge = next(row for row in failures if row["_ng_task_index"] == 0)
    assert judge["response"]["output"][0]["generation_token_ids"] == ([77] if native_tokens else [4, 5])
    assert "reward" not in judge
    assert "response" not in judge["_ng_failure_record"]
    assert captures.read_entries("0-0")
    assert list(read_records(output)) == []
    assert all("response" not in row for row in failures if row["_ng_task_index"] != 0)


async def test_budget_drained_work_resumes_without_consuming_attempts(runner_config, monkeypatch):
    from nemo_gym.rollout_store import RolloutStore

    post = AsyncMock(return_value=FakeResponse(200, {"reward": 0.0, "response": {}}))
    install_fake_server_client(monkeypatch, post)
    runner_config.dispatch_budget_s = 0
    with pytest.raises(RuntimeError, match="no score"):
        await RolloutCollectionHelper().run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    before = RolloutStore.read(output).coverage()
    assert (before["attempts"], before["never_dispatched"]) == (0, 3)
    assert not post.called
    runner_config.resume_from_cache = True
    runner_config.dispatch_budget_s = None
    await RolloutCollectionHelper().run_from_config(runner_config)
    after = RolloutStore.read(output).coverage()
    assert (after["attempts"], after["successful"]) == (3, 3)


async def test_journal_resume_preserves_terminal_timeout_opt_in(runner_config, monkeypatch):
    from nemo_gym.rollout_store import RolloutStore

    async def post(**kwargs):
        if kwargs["json"]["task"] == 0:
            return FakeResponse(200, {"reward": 0.0, "response": {}})
        return FakeResponse(200, {"_ng_failure_class": "timeout_exceeded", "_ng_failure_terminal": True})

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    runner_config.resume_from_cache = True
    client.post.reset_mock()
    await RolloutCollectionHelper().run_from_config(runner_config)
    assert not client.post.called
    runner_config.retry_terminal_timeouts = True
    client.post.side_effect = None
    client.post.return_value = FakeResponse(200, {"reward": 1.0, "response": {}})
    await RolloutCollectionHelper().run_from_config(runner_config)
    assert client.post.call_count == 2
    state = RolloutStore.read(Path(runner_config.output_jsonl_fpath))
    assert (state.coverage()["successful"], state.coverage()["attempts"]) == (3, 5)


@pytest.mark.parametrize("interrupted_migration", [False, True])
async def test_invalid_judge_migration_preserves_journal_identity(runner_config, monkeypatch, interrupted_migration):
    from nemo_gym.rollout_store import RolloutStore

    async def post(**kwargs):
        return FakeResponse(
            200, {"reward": 0.0, "response": {}, "invalid_judge_response": kwargs["json"]["task"] == 1}
        )

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    run_id = RolloutStore.read(output).manifest.run_id
    runner_config.resume_from_cache = True
    runner_config.retry_invalid_judge_responses = True
    if interrupted_migration:
        with monkeypatch.context() as patch:

            def interrupted(*args):
                raise OSError("interrupted migration")

            patch.setattr(collection.os, "replace", interrupted)
            with pytest.raises(OSError, match="interrupted migration"):
                await RolloutCollectionHelper().run_from_config(runner_config)
        # The committed sidecar takes precedence over its exact original copy.
        recovered = RolloutStore.read(output)
        assert (recovered.coverage()["successful"], recovered.coverage()["failed"]) == (2, 1)
        # A marker alone must not make conflicting or foreign payloads acceptable.
        failures = collection.failures_path_for(output)
        original = failures.read_bytes()
        bad = list(read_records(failures))[0] | {"response": {"id": "foreign-answer"}}
        failures.write_text(json.dumps(bad) + "\n")
        with pytest.raises(ConfigError, match="Conflicting outcomes"):
            RolloutStore.read(output)
        failures.write_bytes(original)
    client.post.reset_mock()
    client.post.side_effect = None
    client.post.return_value = FakeResponse(200, {"reward": 1.0, "response": {}})
    await RolloutCollectionHelper().run_from_config(runner_config)
    assert client.post.call_count == 1
    assert client.post.call_args.kwargs["json"]["_ng_attempt_index"] == 1
    recovered = RolloutStore.read(output)
    assert recovered.manifest.run_id == run_id
    assert (recovered.coverage()["successful"], recovered.coverage()["attempts"]) == (3, 4)


async def test_structured_failure_preserves_native_metadata_and_transport_uncertainty(monkeypatch):
    row = failing_row()
    install_fake_server_client(monkeypatch, AsyncMock(side_effect=TimeoutError("no reply")))
    owned_row, failure = await next(RolloutCollectionHelper().run_outcomes([row], run_id="owner"))
    assert failure.run_id == "owner"
    assert (failure.source, failure.delivery, failure.failure.stage) == ("collector", "possibly_delivered", None)
    record = collection._episode_record(
        {
            "task_id": {},
            "failure": {
                "failure_reason": "judge unavailable",
                "terminal": False,
                "failure_kind": "judge_failed",
                "stage": "verification",
                "partial_response": {"output": []},
            },
        }
    )
    native = collection._failure_outcome(owned_row, record, "agent")
    assert (native.source, native.delivery, native.failure.failure_kind, native.failure.stage) == (
        "environment",
        "delivered",
        "judge_failed",
        "verification",
    )
    assert native.failure.failure_reason == "judge unavailable"
    assert "partial_response" not in native.model_dump()["failure"]
    assert record["_ng_failure_partial_response"] == {"output": []}


@pytest.mark.parametrize("field", ["policy_base_url", "policy_api_key"])
def test_shipped_agent_runtime_references_do_not_change_identity(tmp_path, field):
    servers = OmegaConf.to_container(
        OmegaConf.load(
            Path(__file__).parents[2] / "responses_api_agents/anyterminal_agent/configs/anyterminal_claude_code.yaml"
        ),
        resolve=False,
    )
    servers.update(policy_model_name="model", policy_api_key="key-a", policy_base_url="http://a/v1")
    rows = [failing_row() | {"agent_ref": {"name": "anyterminal_claude_code"}}]
    source = tmp_path / "tasks.jsonl"
    source.write_text(json.dumps(rows[0]) + "\n")
    before = RunManifest.create(source, rows, {}, servers)
    assert (
        RunManifest.create(source, rows, {}, servers | {field: "new-location-or-credential"}).config_digest
        == before.config_digest
    )
    assert (
        RunManifest.create(source, rows, {}, servers | {"policy_model_name": "another-model"}).config_digest
        != before.config_digest
    )
    settings = servers["anyterminal_claude_code"]["responses_api_agents"]["anyterminal_agent"]["agent_kwargs"]
    settings["max_turns"] = 10
    assert RunManifest.create(source, rows, {}, servers).config_digest != before.config_digest


@pytest.mark.parametrize("source", ["environment", "collector"])
@pytest.mark.parametrize("alias", [False, True])
async def test_unclassified_structured_failure_is_persisted_and_retried(runner_config, monkeypatch, source, alias):
    from nemo_gym.rollout_store import RolloutStore

    async def post(**kwargs):
        row = kwargs["json"]
        if row["task"] == 0:
            return FakeResponse(200, {"reward": 1.0, "response": {}})
        return FakeResponse(
            200,
            RolloutFailure(
                episode_id=EpisodeId(rollout_id=logical_rollout_id(row)),
                run_id=row["_ng_run_id"],
                source=source,
                delivery="delivered",
                failure=EpisodeFailure(failure_reason="Service unavailable", terminal=False),
            ).model_dump(mode="json"),
        )

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    saved = RolloutStore.read(output)
    assert (saved.coverage()["successful"], saved.coverage()["failed"]) == (1, 2)
    assert all(row["_ng_failure_record"]["failure"].get("failure_kind") is None for row in saved.failures())
    assert all("reward" not in row for row in saved.failures())
    if alias:
        shortcut = output.with_name("shortcut.jsonl")
        shortcut.symlink_to(output)
        runner_config.output_jsonl_fpath = str(shortcut)
    client.post.reset_mock()
    client.post.side_effect = None
    client.post.return_value = FakeResponse(200, {"reward": 0.0, "response": {}})
    runner_config.resume_from_cache = True
    await RolloutCollectionHelper().run_from_config(runner_config)
    assert client.post.call_count == 2
    assert RolloutStore.read(output).coverage()["successful"] == 3
    if alias:
        assert shortcut.is_symlink()
        assert not collection.failures_path_for(shortcut).exists()


@pytest.mark.parametrize("explicit", [False, True])
async def test_direct_invocations_own_run_identity_without_mutating_inputs(monkeypatch, explicit):
    install_fake_server_client(
        monkeypatch, AsyncMock(return_value=FakeResponse(200, {"_ng_failure_class": "judge_failed"}))
    )
    row = failing_row()
    original = dict(row)
    helper = RolloutCollectionHelper()
    first_row, first = await next(helper.run_outcomes([row], run_id="one" if explicit else None))
    second_row, second = await next(helper.run_outcomes([row], run_id="two" if explicit else None))
    assert first.run_id != second.run_id
    assert row == original
    assert first_row["_ng_run_id"] == first.run_id
    assert second_row["_ng_run_id"] == second.run_id
    stamped = row | {"_ng_run_id": "saved"}
    _, preserved = await next(helper.run_outcomes([stamped]))
    _, overridden = await next(helper.run_outcomes([stamped], run_id="new"))
    assert preserved.run_id == "saved" and overridden.run_id == "new"
    assert stamped["_ng_run_id"] == "saved"


@pytest.mark.parametrize("failure_class", ["transient", "permanent", "incomplete"])
async def test_nonjudge_failure_retains_response_only_as_diagnostics(runner_config, monkeypatch, failure_class):
    from nemo_gym.rollout_store import RolloutStore

    response = {
        "id": "actual-answer",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "42"}]}],
    }

    async def post(**kwargs):
        if kwargs["json"]["task"] == 0:
            return FakeResponse(200, {"reward": 1.0, "response": {}})
        return FakeResponse(200, {"reward": 0.0, "response": response, "_ng_failure_class": failure_class})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    failures = RolloutStore.read(Path(runner_config.output_jsonl_fpath)).failures()
    assert len(failures) == 2
    for saved in failures:
        assert saved["_ng_failure_response"] == response
        assert "response" not in saved and "reward" not in saved
        assert "response" not in saved["_ng_failure_record"]


@pytest.mark.parametrize("change", ["model", "verifier", "address"])
async def test_no_serve_resume_uses_running_server_identity(runner_config, monkeypatch, change):
    phase = "A"

    async def post(**kwargs):
        row = kwargs["json"]
        if phase == "A" and row["task"] == 1:
            raise TimeoutError("No saved outcome")
        return FakeResponse(200, {"reward": 1.0, "response": {}, "model_used": phase})

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    client.global_config_dict.my_agent.responses_api_agents.impl.update(
        {"model_server": {"name": "policy"}, "resources_server": {"name": "verifier"}}
    )
    client.global_config_dict.policy = {"responses_api_models": {"openai_model": {"model": "A", "host": "old"}}}
    client.global_config_dict.verifier = {"resources_servers": {"example": {"prompt": "original"}}}
    helper = RolloutCollectionHelper()
    await helper.run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    saved = {path: path.read_bytes() for path in output.parent.iterdir() if path.is_file()}
    phase = "B"
    if change == "model":
        client.global_config_dict.policy.responses_api_models.openai_model.model = "B"
    elif change == "verifier":
        client.global_config_dict.verifier.resources_servers.example.prompt = "changed"
    else:
        client.global_config_dict.policy.responses_api_models.openai_model.host = "new"
    runner_config.resume_from_cache = True
    if change == "address":
        await helper.run_from_config(runner_config)
        assert client.post.await_count == 4
        assert RolloutStore.read(output).coverage()["complete"]
    else:
        with pytest.raises(ConfigError, match="incompatible resolved configuration"):
            await helper.run_from_config(runner_config)
        assert client.post.await_count == 3
        assert all(path.read_bytes() == data for path, data in saved.items())


@pytest.mark.parametrize("missing", ["agent", "environment", "model", "verifier"])
def test_manifest_rejects_missing_required_server_configuration(saved_manifest, missing):
    source, rows, _, config, _, _, _ = saved_manifest
    rows[0]["agent_ref"] = {"name": "agent"}
    servers = {
        "agent": {
            "responses_api_agents": {
                "impl": {
                    "model_server": {"name": "model"},
                    "resources_server": {"type": "resources_servers", "name": "verifier"},
                }
            }
        },
        "model": {"responses_api_models": {"impl": {"model": "model-A"}}},
        "verifier": {"resources_servers": {"impl": {}}},
        "environment": {"environment_servers": {"impl": {"agent_server": {"name": "agent"}}}},
    }
    rows[0]["_ng_environment_server"] = "environment"
    del servers[missing]
    with pytest.raises(ConfigError, match="configuration.*" + missing):
        RunManifest.create(source, rows, config, servers)


@pytest.mark.parametrize("route_failures", [False, True])
@pytest.mark.parametrize("all_unscored", [False, True])
async def test_execute_only_completions_are_saved_and_reused(runner_config, monkeypatch, route_failures, all_unscored):
    runner_config.route_failures_to_sidecar = route_failures

    async def post(**kwargs):
        row = kwargs["json"]
        # Stirrup's execute_only /run contract retains the real generated
        # response and cached deliverables, but deliberately omits reward.
        result = {"response": {"output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]}}
        if all_unscored or row["task"]:
            result.update(execute_only=True, deliverables_dir="/cached/completed", elapsed_seconds=3.0)
        else:
            result["reward"] = 1.0
        return FakeResponse(200, result)

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    helper = RolloutCollectionHelper()
    await helper.run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    store = RolloutStore.read(output)
    coverage = store.coverage()
    assert coverage["successful"] == 3 and coverage["failed"] == 0 and coverage["complete"]
    assert coverage["unscored"] == (3 if all_unscored else 2)
    assert all("reward" not in row for row in store.selected("success") if row.get("execute_only"))
    saved = output.read_bytes()
    runner_config.resume_from_cache = True
    await helper.run_from_config(runner_config)
    assert client.post.await_count == 3 and output.read_bytes() == saved


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"execute_only": True},
        {"response": {}},
        {"execute_only": "true", "response": {}},
        {"execute_only": True, "response": {}, "reward": None},
        {"execute_only": True, "response": {}, "reward": False},
    ],
)
def test_unscored_support_does_not_accept_malformed_results(result):
    with pytest.raises(collection.InvalidRolloutResult):
        collection._normalize_rollout_outcome(failing_row() | {"_ng_run_id": "test"}, result)


def test_documented_stirrup_rerun_toggle_preserves_verified_identity(saved_manifest, monkeypatch):
    source, rows, materialized, config, _, _, path = saved_manifest
    benchmark = OmegaConf.to_container(OmegaConf.load("benchmarks/gdpval/config.yaml"), resolve=False)
    settings = next(
        block["responses_api_agents"]["stirrup_agent"]
        for block in benchmark.values()
        if isinstance(block, dict) and "stirrup_agent" in block.get("responses_api_agents", {})
    )
    servers = {
        "agent": {
            "responses_api_agents": {
                "stirrup_agent": {
                    "rerun_incomplete": settings["rerun_incomplete"],
                    "model_name": "A",
                    "task_options": {"rerun_incomplete": False},
                }
            }
        }
    }
    monkeypatch.setenv("RERUN_INCOMPLETE", "false")
    before = RunManifest.create(source, rows, config, servers)
    before.write(path)
    monkeypatch.setenv("RERUN_INCOMPLETE", "true")
    current = RunManifest.create(source, rows, config, servers)
    assert current.config_digest == before.config_digest
    assert not validate_resume(path, current, materialized).identity_overridden
    servers["agent"]["responses_api_agents"]["stirrup_agent"]["task_options"]["rerun_incomplete"] = True
    assert RunManifest.create(source, rows, config, servers).config_digest != before.config_digest


@pytest.mark.parametrize("direct", [False, True])
async def test_reserved_explicit_id_is_rejected_before_dispatch(runner_config, monkeypatch, direct):
    row = failing_row() | {"_ng_rollout_id": "case-a1"}
    client = install_fake_server_client(
        monkeypatch, AsyncMock(return_value=FakeResponse(200, {"_ng_failure_class": "judge_failed"}))
    )
    helper = RolloutCollectionHelper()
    with pytest.raises(ConfigError, match="reserved attempt suffix"):
        if direct:
            await next(helper.run_outcomes([row], run_id="test"))
        else:
            Path(runner_config.input_jsonl_fpath).write_text(json.dumps(row) + "\n")
            await helper.run_from_config(runner_config)
    client.post.assert_not_called()


async def test_resumed_collector_dispatches_longest_previous_failure_first(runner_config, monkeypatch):
    dispatched = []
    phase = "fail"

    async def post(**kwargs):
        task = kwargs["json"]["task"]
        dispatched.append(task)
        if phase == "fail":
            return FakeResponse(200, {"_ng_failure_class": "agent_run_error", "elapsed_seconds": [5, 20, 10][task]})
        return FakeResponse(200, {"response": {}, "reward": 1.0})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    helper = RolloutCollectionHelper()
    with pytest.raises(RuntimeError, match="None of the 3 dispatched"):
        await helper.run_from_config(runner_config)
    assert dispatched == [0, 1, 2]
    dispatched.clear()
    phase = "success"
    runner_config.resume_from_cache = runner_config.dispatch_longest_first = True
    await helper.run_from_config(runner_config)
    assert dispatched == [1, 2, 0]
    assert RolloutStore.read(Path(runner_config.output_jsonl_fpath)).coverage()["complete"]


@pytest.mark.parametrize("mask_setting", [None, True, False], ids=["default", "strict", "attributed-opt-out"])
@pytest.mark.parametrize("judge_failed", [False, True], ids=["completed", "judge-failure"])
async def test_incomplete_capture_policy_preserves_recovery_accounting(
    runner_config, monkeypatch, tmp_path, mask_setting, judge_failed
):
    from nemo_gym.token_id_capture import ParentResolutionStatus, TokenCaptureStore, stamp_lineage
    from nemo_gym.token_id_capture.records import TokenEntry

    runner_config.limit = 1
    captures = TokenCaptureStore(tmp_path / "tokens")
    settings = {"enabled": True, "all_agents": True, "dir": str(tmp_path / "tokens")}
    if mask_setting is not None:
        settings["mask_incomplete_when_attributed"] = mask_setting
    if judge_failed:
        # Even a masked saved answer from a judge failure must not enter the
        # completed-capture denominator or trip the training quality guard.
        settings.update(max_mask_fraction=0.0, mask_fraction_min_samples=1)
    monkeypatch.setattr(collection, "get_global_config_dict", lambda: {"token_id_capture": settings})
    monkeypatch.setattr(collection, "installed_token_source", lambda: captures)
    output_items = [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "final answer", "annotations": []}],
        }
    ]

    async def post(**kwargs):
        entry = TokenEntry(
            rollout_id="0-0",
            model_call_id="kept-answer",
            model="m",
            prompt_token_ids=[1, 2, 3],
            generation_token_ids=[4, 5],
            generation_log_probs=[-0.1, -0.2],
            output_items=output_items,
            token_item_index=0,
        )
        stamp_lineage(entry, None, parent_resolution=ParentResolutionStatus.ROOT)
        captures.append(entry)
        await captures.mark_incomplete("0-0", "uncommitted-call")
        result = {"response": {"model": "m", "output": output_items}}
        result.update({"_ng_failure_class": "judge_failed"} if judge_failed else {"reward": 1.0})
        return FakeResponse(200, result)

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    if judge_failed:
        with pytest.raises(RuntimeError, match="None of the 1 dispatched rollouts produced a result"):
            await RolloutCollectionHelper().run_from_config(runner_config)
    else:
        await RolloutCollectionHelper().run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    saved_path = collection.failures_path_for(output) if judge_failed else output
    [saved] = read_records(saved_path)
    assert saved.get("mask_sample", False) is (mask_setting is not False)
    assert saved["_ng_token_capture"]["capture_incomplete"] is True
    assert saved["_ng_token_capture"]["terminal_attribution"]["chain"] == "delivered"
    assert saved["response"]["output"][0]["generation_token_ids"] == [4, 5]
    history = RolloutStore.read(output)
    coverage = history.coverage()
    assert coverage["successful"] == int(not judge_failed)
    assert coverage["failed"] == int(judge_failed)
    assert coverage["masked"] == int(not judge_failed and mask_setting is not False)
    assert bool(captures.read_entries("0-0")) is (judge_failed or mask_setting is not False)
    if judge_failed:
        assert "reward" not in saved
        assert list(read_records(output)) == []
        assert len(history.pending(3)) == 1
    else:
        assert saved["reward"] == 1.0
        runner_config.resume_from_cache = True
        client.post.reset_mock()
        await RolloutCollectionHelper().run_from_config(runner_config)
        assert not client.post.called


@pytest.mark.parametrize("latest_outcome", ["failure", "interrupted", "omitted", "suppressed_retry"])
@pytest.mark.parametrize("retain_results", [False, True])
async def test_batch_status_uses_current_journal_attempts(tmp_path, monkeypatch, latest_outcome, retain_results):
    """Collection and offline status cannot revive an older success or failure."""
    from nemo_gym.batch_status import observe_materialized_rows
    from nemo_gym.rollout_collection import RolloutAggregationConfig, RolloutAggregationHelper

    monkeypatch.setattr(collection, "get_global_config_dict", lambda: {})
    source = tmp_path / "inputs.jsonl"
    source.write_text(json.dumps(failing_row() | {"task_source": "my_agent_source"}) + "\n")
    output = tmp_path / "rollouts.jsonl"
    batch_manifest = tmp_path / "batch_manifest.json"
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        batch_manifest_fpath=str(batch_manifest),
        retain_results_in_memory=retain_results,
        disable_aggregation=True,
        disable_health_check=True,
        upload_rollouts=False,
    )
    rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
    observed = observe_materialized_rows(rows)["my_agent"]
    batch_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "members": {
                    "benchmark": {
                        "agent_name": "my_agent",
                        "task_sources": observed.task_sources,
                        "dataset_sha256": observed.dataset_sha256,
                        "expected_task_count": observed.task_count,
                        "expected_rollout_count": observed.rollout_count,
                        "repeat_policy": observed.repeat_policy.model_dump(),
                        "resolved_recipe_sha256": "a" * 64,
                        "metric_keys": [],
                    }
                },
            }
        )
    )
    post = AsyncMock(return_value=FakeResponse(200, {"reward": 1.0, "response": {}}))
    install_fake_server_client(monkeypatch, post)
    await RolloutCollectionHelper().run_from_config(config)
    success = list(read_records(output))[0]
    original_output = output.read_bytes()
    store = RolloutStore.read(output)
    with RolloutStore.start_or_resume(output, lambda: (rows, store.manifest), resume=True) as writer:
        older_failure = rows[0] | {"_ng_run_id": store.manifest.run_id, "_ng_attempt_index": 1}
        writer.record_dispatch(older_failure)
        writer.record_outcome(older_failure | {"_ng_failure_class": "agent_run_error"})
        latest = older_failure | {"_ng_attempt_index": 2}
        writer.record_dispatch(latest)
        if latest_outcome in {"failure", "suppressed_retry"}:
            writer.record_outcome(latest | {"_ng_failure_class": "judge_failed"})
        elif latest_outcome == "omitted":
            writer.record_omission(latest, "No reusable answer")
    assert success["reward"] == 1.0
    post.reset_mock()
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "3")
    resumed = config.model_copy(update={"resume_from_cache": True})
    if latest_outcome == "suppressed_retry":
        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "4")
        post.return_value = FakeResponse(200, {"_ng_no_persist": True})
        with pytest.raises(RuntimeError, match="None of the 1 dispatched"):
            await RolloutCollectionHelper().run_from_config(resumed)
        post.assert_awaited_once()
        post.reset_mock()
    else:
        await RolloutCollectionHelper().run_from_config(resumed)
    expected_failures = {"judge_failed": 1} if latest_outcome == "failure" else {}
    status = json.loads((tmp_path / "batch_status.json").read_text())["members"]["my_agent"]
    assert status["completed_rollout_count"] == 0
    assert status["remaining_rollout_count"] == 1
    assert status["failures_by_class"] == expected_failures
    await RolloutAggregationHelper().run_from_config(
        RolloutAggregationConfig(
            input_glob=str(output),
            output_jsonl_fpath=str(output),
            merge_shards=False,
            batch_manifest_fpath=str(batch_manifest),
            disable_health_check=True,
        )
    )
    offline = json.loads((tmp_path / "batch_status.json").read_text())["members"]["my_agent"]
    assert offline["completed_rollout_count"] == 0
    assert offline["remaining_rollout_count"] == 1
    assert offline["failures_by_class"] == expected_failures
    post.assert_not_awaited()
    assert output.read_bytes() == original_output
