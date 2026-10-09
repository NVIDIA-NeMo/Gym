# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the failure/recovery and checkpoint-controller boundary."""

import asyncio
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
            row["_ng_failure_record"]["failure"]["failure_kind"] == "environment_protocol_violation"
            for row in store.failures()
        )
        assert store.pending(3) == []


@pytest.mark.parametrize("malformed", [None, [], {}, {"reward": float("nan")}])
async def test_bad_reply_is_terminal_and_received_delivery_is_known(monkeypatch, malformed):
    from tests.unit_tests.test_rollout_collection import failing_row

    install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, malformed)))
    _, outcome = await next(collection.RolloutCollectionHelper().run_outcomes([failing_row()]))
    assert outcome.source == "collector" and outcome.delivery == "delivered"
    assert outcome.failure.failure_kind == "environment_protocol_violation" and outcome.failure.terminal


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


@pytest.fixture
def serialized_legacy_failure():
    """Use the actual legacy response model's default JSON serialization."""
    from pydantic import ConfigDict

    from nemo_gym.base_resources_server import BaseVerifyResponse
    from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

    class LegacyFailureResponse(BaseVerifyResponse):
        model_config = ConfigDict(extra="allow")

    def serialize(kind, *, terminal=False):
        result = LegacyFailureResponse(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            response=NeMoGymResponse.model_construct(id="resp-1", output=[]),
            reward=0.0,
            **{"_ng_failure_class": kind, "_ng_failure_terminal": terminal},
        ).model_dump(mode="json")
        assert result["failure_kind"] is None
        return result

    return serialize


@pytest.mark.parametrize(
    "kind", ["timeout_exceeded", "kill_shaped", "reference_missing", "eval_missing", "transport_ineligible"]
)
def test_serialized_legacy_failure_keeps_routing_and_explicit_zero_policy(serialized_legacy_failure, kind):
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_run_id": "run"}
    outcome = collection._normalize_rollout_outcome(row, serialized_legacy_failure(kind))
    assert outcome.failure.failure_kind == kind
    persisted = collection._failure_compatibility_row(outcome)
    assert persisted["_ng_failure_class"] == kind
    assert "reward" not in persisted
    [counted] = collection._counted_failure_rows([persisted], [kind])
    assert counted["reward"] == 0.0
    assert collection._counted_failure_rows([persisted], ["environment_server_failed"]) == []
    assert "reward" not in persisted


async def test_serialized_terminal_timeout_retries_only_with_opt_in(
    runner_config, monkeypatch, serialized_legacy_failure
):
    payload = serialized_legacy_failure("timeout_exceeded", terminal=True)

    async def post(**kwargs):
        result = payload if kwargs["json"]["task"] == 1 else {"response": {}, "reward": 1.0}
        return FakeResponse(200, result)

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    helper = collection.RolloutCollectionHelper()
    await helper.run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    [saved] = RolloutStore.read(output).failures()
    assert saved["_ng_failure_class"] == "timeout_exceeded"
    assert "reward" not in saved
    runner_config.resume_from_cache = True
    client.post.reset_mock()
    await helper.run_from_config(runner_config)
    assert client.post.await_count == 0
    runner_config.retry_terminal_timeouts = True
    client.post = AsyncMock(return_value=FakeResponse(200, {"response": {}, "reward": 1.0}))
    await helper.run_from_config(runner_config)
    assert client.post.await_count == 1
    store = RolloutStore.read(output)
    assert store.coverage()["successful"] == 3
    assert list(read_records(failures_path_for(output))) == [saved]


@pytest.mark.parametrize("second_path", ["alias", "target"])
async def test_fresh_symlink_collector_keeps_exclusive_ownership(runner_config, monkeypatch, second_path):
    output = Path(runner_config.output_jsonl_fpath)
    output.touch()
    alias = output.with_name("latest.jsonl")
    alias.symlink_to(output)
    runner_config.output_jsonl_fpath = str(alias)
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def post(**kwargs):
        calls.append(kwargs["json"]["_ng_run_id"])
        if len(calls) == 1:
            started.set()
            await release.wait()
        return FakeResponse(200, {"response": {}, "reward": 1.0})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    first = asyncio.create_task(collection.RolloutCollectionHelper().run_from_config(runner_config))
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        before = snapshot(output)
        second_config = runner_config.model_copy(
            update={"output_jsonl_fpath": str(alias if second_path == "alias" else output)}
        )
        with pytest.raises(ConfigError, match="Another collector owns this run"):
            await collection.RolloutCollectionHelper().run_from_config(second_config)
        assert snapshot(output) == before
        release.set()
        results = await asyncio.wait_for(first, timeout=10)
        assert len(calls) == len(results) == 3
        assert len(set(calls)) == 1
        assert alias.is_symlink()
        assert RolloutStore.read(alias).selected("success") == results
        assert manifest_path_for(output).exists() and not manifest_path_for(alias).exists()
    finally:
        release.set()
        if not first.done():
            first.cancel()
        await asyncio.gather(first, return_exceptions=True)


@pytest.mark.parametrize(
    "payload",
    [
        {"response": "answer", "reward": 1.0},
        {"response": {"metadata": "custom-metadata"}, "reward": 1.0},
        {"response": ["answer"], "elapsed_seconds": 5},
    ],
)
async def test_native_result_schema_survives_collection_read_and_resume(runner_config, monkeypatch, payload):
    from omegaconf import OmegaConf

    Path(runner_config.input_jsonl_fpath).write_bytes(
        orjson.dumps({"task_id": {"taskset": "native", "task_id": "0"}, "task_input": {}}) + b"\n"
    )
    runner_config.environment_server_routes = {"native": "environment"}

    async def post(**kwargs):
        body = kwargs["json"]
        return FakeResponse(
            200,
            {
                "episode_id": body["episode_id"],
                "task_id": body["task"]["task_id"],
                "result": payload,
            },
        )

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    client.global_config_dict = OmegaConf.create({"environment": {"environment_servers": {"custom": {}}}})
    results = await collection.RolloutCollectionHelper().run_from_config(runner_config)
    assert results[0]["response"] == payload["response"]
    output = Path(runner_config.output_jsonl_fpath)
    assert RolloutStore.read(output).selected("success") == results
    saved = output.read_bytes()
    runner_config.resume_from_cache = True
    assert await collection.RolloutCollectionHelper().run_from_config(runner_config) == results
    assert client.post.await_count == 1
    assert output.read_bytes() == saved


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"response": "answer", "elapsed_seconds": 2}, 2.0),
        ({"response": {"metadata": ["custom"]}}, None),
        ({"response": {"metadata": {"elapsed_seconds": "3.5"}}}, 3.5),
        ({"elapsed_seconds": 1, "response": {"metadata": {"elapsed_seconds": 8}}}, 1.0),
        ({"elapsed_seconds": "invalid", "response": {"metadata": {"elapsed_seconds": 8}}}, 8.0),
    ],
)
def test_optional_elapsed_hints_allow_environment_owned_response_shapes(record, expected):
    from nemo_gym.rollout_recovery import observed_elapsed

    assert observed_elapsed(record) == expected


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("allow_unsafe", [False, True])
def test_missing_companions_never_erase_run_tagged_outcomes(prepared_run, failed, allow_unsafe):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.allocate_attempt(row)
        payload = {"_ng_failure_class": "judge_failed"} if failed else {"reward": 1.0, "response": {}}
        store.record_outcome(row | payload)
    manifest_path_for(output).unlink()
    materialized_path_for(output).unlink()
    before = snapshot(output)
    prepare.reset_mock()
    with pytest.raises(ConfigError, match="lost their manifest"):
        RolloutStore.start_or_resume(output, prepare, resume=True, allow_unsafe=allow_unsafe)
    assert snapshot(output) == before
    prepare.assert_not_called()


def test_interrupted_fresh_replacement_requires_repeating_fresh_command(prepared_run, monkeypatch):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.allocate_attempt(row)
        store.record_outcome(row | {"reward": 1.0, "response": {}})
    old_run_id = store.manifest.run_id
    rows, _ = prepare()
    changed = [row | {"task": "new"} for row in rows]
    source = output.with_name("source.jsonl")
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in changed))

    def prepare_new():
        return changed, RunManifest.create(source, changed, {}, {"agent": {"responses_api_agents": {"impl": {}}}})

    def interrupted_write(*args):
        raise OSError("interrupted before publishing new manifest")

    with monkeypatch.context() as patch:
        patch.setattr(RunManifest, "write", interrupted_write)
        with pytest.raises(OSError, match="interrupted"):
            RolloutStore.start_or_resume(output, prepare_new, resume=False)
    before = snapshot(output)
    for unsafe in (False, True):
        with pytest.raises(ConfigError, match="materialized inputs"):
            RolloutStore.start_or_resume(output, prepare_new, resume=True, allow_unsafe=unsafe)
        assert snapshot(output) == before
    fresh = RolloutStore.start_or_resume(output, prepare_new, resume=False)
    assert fresh.manifest.run_id != old_run_id
    assert fresh.coverage()["attempts"] == 0
    assert [row["task"] for row in fresh.pending(3)] == ["new", "new"]


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_gateway_failure_does_not_prove_environment_delivery(status):
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0, "_ng_run_id": "run"}
    failure = collection._failure_outcome(
        row, {"_ng_failure_class": "agent_run_error", "_ng_failure_http_status": status}, "request"
    )
    assert failure.source == "collector"
    assert failure.delivery == "possibly_delivered"


async def test_producer_error_is_preserved_in_persisted_sidecar(runner_config, monkeypatch):
    async def post(**kwargs):
        if kwargs["json"]["task"] == 1:
            return FakeResponse(
                200, {"_ng_failure_class": "agent_run_error", "error": "Sandbox unavailable: original evidence"}
            )
        return FakeResponse(200, {"response": {}, "reward": 1.0})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await collection.RolloutCollectionHelper().run_from_config(runner_config)
    [failure] = RolloutStore.read(Path(runner_config.output_jsonl_fpath)).failures()
    assert failure["error"] == "Sandbox unavailable: original evidence"
    assert failure["_ng_failure_record"]["failure"]["failure_reason"] == "Agent reported a no-result failure"
    assert "reward" not in failure


@pytest.mark.parametrize("unsafe", [False, True])
def test_legacy_alias_companions_never_erase_saved_results(prepared_run, unsafe):
    output, prepare = prepared_run
    output.write_text('{"_ng_task_index":0,"_ng_rollout_index":0,"reward":1}\n')
    alias = output.with_name("latest.jsonl")
    alias.symlink_to(output)
    materialized_path_for(alias).write_bytes(prepare()[1].model_dump_json().encode())
    before = snapshot(output)
    with pytest.raises(ConfigError, match="beside the rollout alias"):
        RolloutStore.start_or_resume(alias, prepare, resume=True, allow_unsafe=unsafe)
    with pytest.raises(ConfigError, match="beside the rollout alias"):
        collection._expand_input_glob(str(alias))
    assert snapshot(output) == before


@pytest.mark.parametrize("kind", ["terminal", "omitted", "exhausted", "retryable", "unknown"])
def test_cli_retryability_matches_store_pending_work(prepared_run, kind):
    from nemo_gym.cli.eval import _check_saved_completion

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row, unfinished = store.pending(3)
        store.record_dispatch(row)
        store.record_outcome(row | {"reward": 1.0, "response": {}})
        if kind != "unknown":
            for _ in range(3 if kind == "exhausted" else 1):
                store.allocate_attempt(unfinished)
                store.record_outcome(
                    unfinished
                    | {
                        "_ng_failure_class": "skipped" if kind == "omitted" else "agent_run_error",
                        "_ng_failure_terminal": kind in {"terminal", "omitted"},
                    }
                )
        assert bool(store.pending(3)) is (kind in {"retryable", "unknown"})
    with pytest.raises(IncompleteEvaluationError) as error:
        _check_saved_completion(output)
    assert error.value.exit_code == (75 if kind in {"retryable", "unknown"} else 76)


async def test_aggregate_without_merge_preserves_source_coverage(prepared_run, monkeypatch):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False):
        pass
    before = coverage_path_for(output).read_bytes()
    monkeypatch.setattr(collection.RolloutCollectionHelper, "_call_aggregate_metrics", AsyncMock(return_value=None))
    await collection.RolloutAggregationHelper().run_from_config(
        collection.RolloutAggregationConfig(
            input_glob=str(output),
            output_jsonl_fpath=str(output),
            merge_shards=False,
            disable_health_check=True,
        )
    )
    assert coverage_path_for(output).read_bytes() == before
    assert output.with_name("rollouts_aggregate_coverage.json").exists()


def test_batch_reservations_are_durable_before_dispatch(prepared_run, monkeypatch):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        rows = store.pending(3)
        original_write = RunManifest.write
        writes = []

        def write(manifest, path):
            writes.append(dict(manifest.next_attempt))
            original_write(manifest, path)

        monkeypatch.setattr(RunManifest, "write", write)
        store.record_dispatches(rows)
        assert len(writes) == 1
        assert writes[0] == {"0-0": 1, "1-0": 1}
        assert len(RolloutStore.read(output).pending(3)) == 2
        store.record_dispatches(rows)
        assert len(writes) == 1
        assert all(row["_ng_attempt_index"] == 0 for row in rows)


def test_failed_batch_reservation_changes_neither_memory_nor_disk(prepared_run, monkeypatch):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        rows = store.pending(3)
        original_rows = [dict(row) for row in rows]
        before = snapshot(output)

        def fail(*args):
            raise OSError("disk full")

        monkeypatch.setattr(RunManifest, "write", fail)
        with pytest.raises(OSError, match="disk full"):
            store.record_dispatches(rows)
        assert store.manifest.next_attempt == {}
        assert rows == original_rows
        assert snapshot(output) == before


@pytest.mark.parametrize("newer", ["none", "interrupted", "failed", "succeeded"])
def test_health_only_accepts_already_selected_rows(prepared_run, newer):
    from scripts.harness_conformance.runner import inspect_episode
    from scripts.harness_conformance.scenarios import SCENARIOS

    from nemo_gym.rollout_health import JournalHealthUnavailable, run_health_checks

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row, failed = store.pending(3)
        store.record_dispatches([row, failed])
        store.record_outcome(row | {"reward": 1.0, "response": {}})
        store.record_outcome(failed | {"_ng_failure_class": "judge_failed"})
        if newer != "none":
            store.allocate_attempt(row)
            if newer != "interrupted":
                store.record_outcome(
                    row
                    | (
                        {"_ng_failure_class": "agent_run_error"}
                        if newer == "failed"
                        else {"reward": 0.0, "response": {}}
                    )
                )
    before = output.read_bytes(), failures_path_for(output).read_bytes()
    if newer == "none":
        run_health_checks([output, failures_path_for(output)], workers=1)
        assert (output.parent / "quality_summary.json").exists()
        # The production conformance reader must reach reporting, even when a
        # minimal synthetic result fails its unrelated capability checks.
        report = inspect_episode(SCENARIOS[0], output.parent, {})
        assert report is not None
    else:
        with pytest.raises(JournalHealthUnavailable, match="superseded outcomes"):
            run_health_checks(output, workers=1)
    assert before == (output.read_bytes(), failures_path_for(output).read_bytes())


async def test_ordinary_reverification_reads_fresh_manifest_without_changing_source(prepared_run, monkeypatch):
    import nemo_gym.rollout_reverification as reverify

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for row in store.pending(3):
            store.record_dispatch(row)
            store.record_outcome(row | {"reward": 0.0, "response": {"answer": "saved"}})
    before = snapshot(output)
    monkeypatch.setattr(reverify, "_guard_reverify_mode", AsyncMock(return_value=None))
    seen = []

    async def verify(row):
        seen.append(row)
        return row, {"reward": 1.0, "response": row["response"]}

    monkeypatch.setattr(
        reverify, "_run_verification_payloads", lambda payloads, **kwargs: [verify(row) for row in payloads]
    )
    await reverify.RolloutReverificationHelper().run_from_config(
        reverify.RolloutReverificationConfig(
            materialized_inputs_jsonl_fpath=str(materialized_path_for(output)),
            rollouts_jsonl_fpath=str(output),
            output_jsonl_fpath=str(output.with_name("rescored.jsonl")),
            disable_aggregation=True,
            upload_rollouts=False,
        )
    )
    assert len(seen) == 2 and all(row["response"] == {"answer": "saved"} for row in seen)
    assert all((output.parent / name).read_bytes() == content for name, content in before.items())
    assert all(row["reward"] == 1.0 for row in read_records(output.with_name("rescored.jsonl")))


async def test_production_dispatch_batches_reservations_before_http(runner_config, monkeypatch):
    output = Path(runner_config.output_jsonl_fpath)
    runner_config.num_samples_in_parallel = 3
    writes = []
    original = RunManifest.write

    def write(manifest, path):
        writes.append(dict(manifest.next_attempt))
        original(manifest, path)

    monkeypatch.setattr(RunManifest, "write", write)

    async def post(**kwargs):
        row = kwargs["json"]
        saved = RunManifest.model_validate_json(manifest_path_for(output).read_bytes())
        assert saved.next_attempt[collection.logical_rollout_id(row)] > row["_ng_attempt_index"]
        return FakeResponse(200, {"response": {}, "reward": 1.0})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await collection.RolloutCollectionHelper().run_from_config(runner_config)
    assert writes == [{}, {"0-0": 1, "1-0": 1, "2-0": 1}]


async def test_invalid_reply_without_routing_explains_how_to_continue(runner_config, monkeypatch):
    runner_config.route_failures_to_sidecar = False
    install_fake_server_client(monkeypatch, AsyncMock(return_value=FakeResponse(200, {"reward": 1.0})))
    with pytest.raises(collection.InvalidRolloutResult) as error:
        await collection.RolloutCollectionHelper().run_from_config(runner_config)
    assert any("+route_failures_to_sidecar=true" in note for note in error.value.__notes__)


async def test_reservation_write_failure_prevents_http_dispatch(runner_config, monkeypatch):
    original = RunManifest.write

    def write(manifest, path):
        if manifest.next_attempt:
            raise OSError("reservation disk unavailable")
        original(manifest, path)

    monkeypatch.setattr(RunManifest, "write", write)
    client = install_fake_server_client(monkeypatch, AsyncMock())
    with pytest.raises(OSError, match="reservation disk unavailable"):
        await collection.RolloutCollectionHelper().run_from_config(runner_config)
    client.post.assert_not_awaited()
    output = Path(runner_config.output_jsonl_fpath)
    assert RunManifest.model_validate_json(manifest_path_for(output).read_bytes()).next_attempt == {}
    assert output.read_bytes() == b""


def test_duplicate_batch_identity_is_rejected_before_persistence(prepared_run):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(3)[0]
        before = snapshot(output)
        with pytest.raises(ConfigError, match="Duplicate rollout"):
            store.record_dispatches([row, dict(row)])
        assert snapshot(output) == before
        assert store.manifest.next_attempt == {}
