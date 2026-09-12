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
import pickle
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

import nemo_gym.rollout_collection as collection
import nemo_gym.rollout_reverification as reverification
from nemo_gym.config_types import ConfigError
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
from tests.unit_tests.test_rollout_collection import FakeResponse, failing_row, http_error, install_fake_server_client


def echo_failure(connection, failure):
    """Exercise the spawn/pickle boundary used by process-based callers."""
    connection.send(failure)
    connection.close()


def test_failure_has_no_generation_fields_and_round_trips():
    failure = RolloutFailure(
        rollout_id="task-42",
        attempt_index=2,
        failure_kind="agent_request_failed",
        stage="request",
        failure_reason="x" * 9000,
        response_body="y" * 9000,
    )
    assert failure.attempt_id == "task-42-a2"
    assert len(failure.failure_reason) == len(failure.response_body) == 2000
    assert RolloutFailure.model_validate_json(failure.model_dump_json()) == failure
    assert pickle.loads(pickle.dumps(failure)) == failure
    for field in ("reward", "response", "messages", "tokens"):
        with pytest.raises(ValidationError):
            RolloutFailure.model_validate(failure.model_dump() | {field: 0})


def test_failure_survives_spawned_process():
    failure = RolloutFailure(
        rollout_id="task-42",
        failure_kind="transport_timeout",
        stage="request",
        failure_reason="Timed out",
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
    assert original is row
    assert isinstance(outcome, RolloutFailure)
    assert outcome.stage == "result"
    assert outcome.exception_type == "InvalidRolloutResult"
    assert outcome.rollout_id
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
    row = failing_row() | {"_ng_rollout_id": "stable-task", "_ng_attempt_index": 2}
    failure = RolloutFailure(
        rollout_id="stable-task",
        attempt_index=2,
        failure_kind="judge_failed",
        stage="verifier",
        failure_reason="Judge unavailable",
    )
    payload = failure.model_dump()
    if wrong_identity == "rollout_id":
        payload["rollout_id"] = "another-task"
    elif wrong_identity == "attempt_index":
        payload["attempt_index"] = 1
    elif wrong_identity == "run_id":
        payload["run_id"] = "another-run"
    install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, payload)))
    original, outcome = await next(RolloutCollectionHelper().run_outcomes([row]))
    assert original is row
    if wrong_identity is None:
        assert outcome == failure
    else:
        assert outcome.rollout_id == failure.rollout_id
        assert outcome.attempt_index == failure.attempt_index
        assert outcome.exception_type == "InvalidRolloutResult"
        assert outcome.stage == "result"


async def test_legacy_judge_placeholder_is_converted_only_for_typed_callers(monkeypatch):
    legacy = {
        "reward": 0.0,
        "response": {"output": []},
        "_ng_failure_class": "judge_failed",
        "failure_reason": "Judge unavailable",
    }
    install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, legacy)))
    _, raw = await next(RolloutCollectionHelper().run_examples([failing_row()]))
    assert raw is legacy
    _, outcome = await next(RolloutCollectionHelper().run_outcomes([failing_row()]))
    assert isinstance(outcome, RolloutFailure)
    assert outcome.failure_kind == "judge_failed"
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
    config = config | {"resume_from_cache": True, "num_samples_in_parallel": 10}
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


@pytest.mark.parametrize("route_failures", [False, True])
@pytest.mark.parametrize("append", [False, True])
async def test_collected_judge_failure_can_be_reverified_without_inference(
    runner_config, monkeypatch, route_failures, append
):
    runner_config.route_failures_to_sidecar = route_failures
    generated_response = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "42"}]}]}

    async def post(**kwargs):
        row = kwargs["json"]
        if kwargs["url_path"] == "/verify":
            assert row["task"] == 1 and row["response"] == generated_response
            return FakeResponse(200, {"reward": 1.0, "response": row["response"]})
        assert kwargs["url_path"] == "/run"
        if row["task"] == 1:
            return FakeResponse(
                200,
                {
                    "_ng_failure_class": "judge_failed",
                    "failure_reason": "Judge unavailable",
                    "reward": 0.0,
                    "response": generated_response,
                },
            )
        if row["task"] == 2:
            return FakeResponse(
                200,
                RolloutFailure(
                    rollout_id=logical_rollout_id(row),
                    failure_kind="judge_failed",
                    stage="verifier",
                    failure_reason="No generation saved",
                ).model_dump(),
            )
        return FakeResponse(200, {"reward": 0.0, "response": {"output": []}})

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    successes = list(read_records(output))
    failures = {row["_ng_task_index"]: row for row in read_records(collection.failures_path_for(output))}
    assert failures[1]["response"] == generated_response
    assert "reward" not in failures[1]
    assert "response" not in failures[2]
    for row in failures.values():
        failure = RolloutFailure.model_validate(row["_ng_failure_record"])
        assert "response" not in failure.model_dump() and "reward" not in failure.model_dump()

    # Exercise the real sidecar reader, materialized-input join, payload builder,
    # verifier dispatch, and output writer; only the HTTP boundary is replaced.
    monkeypatch.setattr(reverification, "setup_server_client", lambda: client)
    monkeypatch.setattr(reverification, "_build_agent_to_resources_server_mapping", lambda _: {"my_agent": "rs"})
    monkeypatch.setattr(reverification, "raise_for_status", collection.raise_for_status)
    monkeypatch.setattr(reverification, "get_response_json", collection.get_response_json)
    monkeypatch.setattr(reverification, "get_exporters", list)
    config = reverification.RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(runner_config.materialized_jsonl_fpath),
        rollouts_jsonl_fpath=str(output),
        output_jsonl_fpath=str(output if append else output.with_name("reverified.jsonl")),
        judge_failed_only=True,
        append=append,
        disable_aggregation=True,
    )
    with pytest.warns(UserWarning, match="without a saved response"):
        returned = await reverification.RolloutReverificationHelper().run_from_config(config)
    by_task = {row["_ng_task_index"]: row for row in returned}
    assert by_task[0] == successes[0]
    assert by_task[1]["reward"] == 1.0 and by_task[1]["response"] == generated_response
    assert set(by_task) == {0, 1}
    assert [call.kwargs["url_path"] for call in client.post.await_args_list].count("/run") == 3
    assert [call.kwargs["url_path"] for call in client.post.await_args_list].count("/verify") == 1
    if append:
        from nemo_gym.rollout_store import RolloutStore

        recovered = RolloutStore.read(output)
        assert recovered.selected("success") == returned
        assert by_task[1]["_ng_attempt_index"] == 1
        assert recovered.coverage()["attempts"] == 4
        with pytest.warns(UserWarning, match="without a saved response"):
            assert await reverification.RolloutReverificationHelper().run_from_config(config) == returned
        assert [call.kwargs["url_path"] for call in client.post.await_args_list].count("/verify") == 1


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


async def test_reported_kill_shaped_failures_consume_bounded_attempts(runner_config, monkeypatch):
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
    assert json.loads(coverage_path_for(output).read_text())["attempts"] == 5


async def test_cancelled_reverify_append_stops_requests_before_closing_journal(runner_config, monkeypatch):
    from nemo_gym.rollout_store import RolloutStore

    started, stopped = asyncio.Event(), asyncio.Event()
    verify_requests = []

    async def post(**kwargs):
        row = kwargs["json"]
        if kwargs["url_path"] == "/verify":
            verify_requests.append(row)
            started.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()
        result = {"reward": 0, "response": {}}
        if row["task"] != 0:
            result["_ng_failure_class"] = "judge_failed"
        return FakeResponse(200, result)

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    monkeypatch.setattr(reverification, "setup_server_client", lambda: client)
    monkeypatch.setattr(reverification, "_build_agent_to_resources_server_mapping", lambda _: {"my_agent": "rs"})
    monkeypatch.setattr(reverification, "raise_for_status", collection.raise_for_status)
    monkeypatch.setattr(reverification, "get_response_json", collection.get_response_json)
    config = reverification.RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(runner_config.materialized_jsonl_fpath),
        rollouts_jsonl_fpath=runner_config.output_jsonl_fpath,
        output_jsonl_fpath=runner_config.output_jsonl_fpath,
        judge_failed_only=True,
        append=True,
        num_samples_in_parallel=1,
        disable_aggregation=True,
    )
    task = asyncio.create_task(reverification.RolloutReverificationHelper().run_from_config(config))
    await asyncio.wait_for(started.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set() and len(verify_requests) == 1
    coverage = RolloutStore.read(Path(runner_config.output_jsonl_fpath)).coverage()
    assert (coverage["attempts"], coverage["successful"], coverage["failed"], coverage["unknown"]) == (4, 1, 1, 1)


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
        "coverage/failed": 1,
        "coverage/omitted": 1,
        "coverage/unknown": 0,
    }
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
                rollout_id=logical_rollout_id(row),
                failure_kind="judge_failed",
                stage="verifier",
                failure_reason="Judge unavailable",
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
        assert row["_ng_failure_message"] == row["_ng_failure_record"]["failure_reason"] == row["error"]
