# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the failure/recovery and checkpoint-controller boundary."""

import multiprocessing
import os
import signal
from pathlib import Path
from unittest.mock import AsyncMock

import orjson
import pytest

import nemo_gym.rollout_collection as collection
from nemo_gym.cli.utils import exit_cleanly_on_config_error
from nemo_gym.config_types import ConfigError
from nemo_gym.episode_types import EpisodeFailure, EpisodeId
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_outcomes import RolloutFailure
from nemo_gym.rollout_records import coverage_path_for, journal_path_for, materialized_path_for, read_records
from nemo_gym.rollout_recovery import IncompleteEvaluationError, RunManifest, manifest_path_for, run_lock
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client
from tests.unit_tests.test_rollout_recovery import runner_config  # noqa: F401
from tests.unit_tests.test_rollout_store import prepared_run, snapshot  # noqa: F401


@pytest.mark.parametrize("kind", [None, "judge_failed"])
def test_native_unclassified_failure_preserves_producer_metadata(kind):
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_run_id": "run"}
    result = collection._episode_record(
        {
            "task_id": {},
            "failure": {
                "failure_kind": kind,
                "failure_reason": "Unavailable",
                "terminal": False,
            },
        }
    )
    failure = collection._failure_outcome(row, result, "agent")
    assert failure.failure.failure_kind == kind
    assert failure.sidecar_failure_class == (kind or "environment_server_failed")


@pytest.mark.parametrize("routing", [False, True])
@pytest.mark.parametrize("native", [False, True])
async def test_invalid_http_success_never_completes_evaluation(runner_config, monkeypatch, routing, native):
    runner_config.route_failures_to_sidecar = routing
    if native:
        Path(runner_config.input_jsonl_fpath).write_text(
            orjson.dumps(
                {
                    "task_id": {"taskset": "native", "task_id": "0"},
                    "task_input": {},
                }
            ).decode()
            + "\n"
        )
        runner_config.environment_server_routes = {"native": "environment"}
    client = install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, {"error": "bad"})))
    if native:
        from omegaconf import OmegaConf

        client.global_config_dict = OmegaConf.create({"environment": {"environment_servers": {"custom": {}}}})
    with pytest.raises((collection.InvalidRolloutResult, IncompleteEvaluationError, RuntimeError)):
        await collection.RolloutCollectionHelper().run_from_config(runner_config)
    store = RolloutStore.read(Path(runner_config.output_jsonl_fpath))
    assert store.coverage()["successful"] == 0
    if routing:
        assert all(
            row["_ng_failure_record"]["failure"]["failure_kind"] == "protocol_violation" for row in store.failures()
        )
        assert store.pending(3) == []


@pytest.mark.parametrize("malformed", [None, [], {}, {"reward": float("nan")}])
async def test_bad_reply_is_terminal_and_received_delivery_is_known(monkeypatch, malformed):
    from tests.unit_tests.test_rollout_collection import failing_row

    install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, malformed)))
    _, outcome = await next(collection.RolloutCollectionHelper().run_outcomes([failing_row()]))
    assert outcome.source == "collector" and outcome.delivery == "delivered"
    assert outcome.failure.failure_kind == "protocol_violation" and outcome.failure.terminal


@pytest.mark.parametrize("routing", [False, True])
async def test_builtin_native_judge_failure_preserves_answer(runner_config, monkeypatch, routing):
    from omegaconf import OmegaConf

    Path(runner_config.input_jsonl_fpath).write_text(
        orjson.dumps(
            {
                "task_id": {"taskset": "native", "task_id": "0"},
                "task_input": {},
            }
        ).decode()
        + "\n"
    )
    runner_config.environment_server_routes = {"native": "environment"}
    runner_config.route_failures_to_sidecar = routing
    answer = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "42"}]}]}

    async def post(**kwargs):
        body = kwargs["json"]
        return FakeResponse(
            200,
            {
                "episode_id": body["episode_id"],
                "task_id": body["task"]["task_id"],
                "result": {
                    "reward": 0.0,
                    "response": answer,
                    "mask_sample": True,
                    "failure_kind": "judge_failed",
                    "failure_reason": "Judge unavailable",
                    "_ng_failure_class": "judge_failed",
                    "_ng_failure_judge_error": "Judge unavailable",
                },
            },
        )

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    client.global_config_dict = OmegaConf.create({"environment": {"environment_servers": {"custom": {}}}})
    with pytest.raises(RuntimeError, match="None of the"):
        await collection.RolloutCollectionHelper().run_from_config(runner_config)
    [saved] = RolloutStore.read(Path(runner_config.output_jsonl_fpath)).failures()
    assert saved["response"] == answer and "reward" not in saved
    assert saved["_ng_failure_record"]["failure"]["failure_kind"] == "judge_failed"


@pytest.mark.parametrize("field", ["error", "error_message", "agent_error", "grading_notes"])
def test_producer_specific_diagnostics_are_not_failure_reasons(field):
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_run_id": "run"}
    result = {"_ng_failure_class": "judge_failed", field: "A grader comment"}
    outcome = collection._failure_outcome(row, result, "agent")
    assert outcome.failure.failure_reason == "Agent reported a no-result failure"
    assert collection._failure_diagnostics(result)[field] == "A grader comment"


@pytest.mark.parametrize("delivery", ["not_sent", "possibly_delivered", "delivered"])
@pytest.mark.parametrize("terminal", [False, True])
def test_only_delivered_nonterminal_failures_spend_budget(prepared_run, delivery, terminal):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.allocate_attempt(row)
        failure = RolloutFailure(
            episode_id=EpisodeId(rollout_id="0-0"),
            run_id=store.manifest.run_id,
            source="collector",
            delivery=delivery,
            failure=EpisodeFailure(failure_reason="Failure", terminal=terminal, failure_kind="transport_timeout"),
        )
        store.record_outcome(row | collection._failure_compatibility_row(failure))
    read = RolloutStore.read(output)
    assert read.coverage()["counted_failures"] == int(delivery != "not_sent" and not terminal)
    assert any(row["_ng_task_index"] == 0 for row in read.pending(1)) == (not terminal and delivery == "not_sent")


def test_interrupted_outcome_indexing_keeps_complete_payload(prepared_run, monkeypatch):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        result = row | {"reward": 0.0, "response": {}}

        def fail(*args, **kwargs):
            raise OSError("interrupted after payload flush")

        monkeypatch.setattr(store._state, "outcome", fail)
        with pytest.raises(OSError, match="interrupted"):
            store.record_outcome(result)
    assert RolloutStore.read(output).selected("success") == [result]
    assert not journal_path_for(output).exists()


@pytest.mark.parametrize("original_error", [False, True])
def test_coverage_write_does_not_mask_collection_error(prepared_run, monkeypatch, caplog, original_error):
    output, prepare = prepared_run
    store = RolloutStore.start_or_resume(output, prepare, resume=False)

    def fail(**kwargs):
        raise OSError("coverage disk full")

    with pytest.raises(ValueError if original_error else OSError, match="primary" if original_error else "coverage"):
        with store:
            monkeypatch.setattr(store, "write_coverage", fail)
            if original_error:
                raise ValueError("primary failure")
    if original_error:
        assert "Could not write coverage" in caplog.text


@pytest.mark.parametrize("content", [b"{", b'{"schema_version":999}'])
def test_bad_manifest_reports_path(prepared_run, content):
    output, prepare = prepared_run
    RolloutStore.start_or_resume(output, prepare, resume=False)
    manifest_path_for(output).write_bytes(content)
    with pytest.raises(ConfigError, match=str(manifest_path_for(output))):
        RolloutStore.read(output)


@pytest.mark.parametrize(
    "companion",
    [
        lambda path: path,
        manifest_path_for,
        journal_path_for,
        materialized_path_for,
        failures_path_for,
        coverage_path_for,
    ],
)
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
async def test_merge_never_overwrites_source_artifacts(prepared_run, monkeypatch, companion, alias):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"reward": 0.0, "response": {}})
    target = companion(output)
    if not target.exists():
        target.write_bytes(b"preserve")
    destination = target
    if alias != "direct":
        destination = output.with_name("alias.jsonl")
        if alias == "symlink":
            destination.symlink_to(target)
        else:
            destination.hardlink_to(target)
    before = snapshot(output)
    monkeypatch.setattr(collection.RolloutCollectionHelper, "_call_aggregate_metrics", AsyncMock())
    with pytest.raises(ConfigError, match="source rollout artifact"):
        await collection.RolloutAggregationHelper().run_from_config(
            collection.RolloutAggregationConfig(
                input_glob=str(output),
                output_jsonl_fpath=str(destination),
                disable_health_check=True,
            )
        )
    assert snapshot(output) == before


def reserve_then_die(output, pipe):
    with run_lock(output):
        manifest = RunManifest.model_validate_json(manifest_path_for(output).read_bytes())
        from nemo_gym.rollout_records import RolloutRecords

        with RolloutStore(output, RolloutRecords.load(output, manifest)) as store:
            row = store.pending(3)[0]
            store.allocate_attempt(row)
            pipe.send(row["_ng_attempt_index"])
            os.kill(os.getpid(), signal.SIGKILL)


def test_process_death_releases_lock_and_preserves_allocator(prepared_run):
    output, prepare = prepared_run
    RolloutStore.start_or_resume(output, prepare, resume=False)
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe()
    process = context.Process(target=reserve_then_die, args=(output, send))
    process.start()
    send.close()
    try:
        assert receive.poll(30)
        assert receive.recv() == 0
        process.join(30)
        assert process.exitcode == -signal.SIGKILL
        with run_lock(output):
            with RolloutStore.start_or_resume(output, prepare, resume=True) as store:
                row = store.pending(1)[0]
                assert store.allocate_attempt(row) == 1
                assert store.coverage()["counted_failures"] == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        receive.close()


@pytest.mark.parametrize("shared_checkpoint", [False, True])
def test_two_controllers_cannot_own_the_same_run(tmp_path, shared_checkpoint):
    output = tmp_path / "out.jsonl"
    checkpoint = tmp_path / "checkpoints" if shared_checkpoint else None
    second = tmp_path / "other.jsonl" if shared_checkpoint else output
    with run_lock(output, checkpoint_dir=checkpoint):
        with pytest.raises(ConfigError, match="Another collector owns"):
            with run_lock(second, checkpoint_dir=checkpoint):
                pytest.fail("Concurrent writer admitted")
    with run_lock(second, checkpoint_dir=checkpoint):
        pass


def test_incomplete_evaluation_has_distinct_cli_exit_code():
    @exit_cleanly_on_config_error
    def interrupted():
        raise IncompleteEvaluationError("Work remains; partial results are saved")

    with pytest.raises(SystemExit) as error:
        interrupted()
    assert error.value.code == 75


def test_output_alias_uses_the_same_lock(tmp_path):
    output = tmp_path / "out.jsonl"
    output.touch()
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(output)
    with run_lock(output):
        with pytest.raises(ConfigError, match="Another collector owns"):
            with run_lock(alias):
                pytest.fail("Alias bypassed run ownership")


@pytest.mark.parametrize("field", ["_ng_run_id", "_ng_attempt_index", "_ng_task_index"])
def test_indexed_read_detects_in_place_identity_rewrite(prepared_run, field):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"reward": 0.0, "response": {}})
    reader = RolloutStore.read(output)
    before_inode = output.stat().st_ino
    [payload] = read_records(output)
    payload[field] = "x" * len(payload[field]) if field == "_ng_run_id" else 9
    output.write_bytes(orjson.dumps(payload) + b"\n")
    assert output.stat().st_ino == before_inode
    with pytest.raises(ConfigError, match="identity changed"):
        reader.selected("success")


def test_cli_incomplete_status_after_saved_budget_drain(tmp_path):
    from nemo_gym.cli.eval import _check_saved_completion

    output = tmp_path / "out.jsonl"
    coverage_path_for(output).write_bytes(orjson.dumps({"complete": False, "successful": 2, "expected": 3}))
    with pytest.raises(IncompleteEvaluationError, match="2/3"):
        _check_saved_completion(output)
    coverage_path_for(output).write_bytes(orjson.dumps({"complete": True, "successful": 3, "expected": 3}))
    _check_saved_completion(output)
