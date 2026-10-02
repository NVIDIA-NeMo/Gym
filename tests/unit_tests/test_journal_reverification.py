# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

import nemo_gym.rollout_collection as collection
import nemo_gym.rollout_reverification as reverification
from nemo_gym.config_types import ConfigError
from nemo_gym.episode_types import EpisodeFailure, EpisodeId
from nemo_gym.rollout_collection import RolloutCollectionHelper
from nemo_gym.rollout_journal import (
    journal_path_for,
    logical_rollout_id,
    materialized_path_for,
    read_records,
)
from nemo_gym.rollout_outcomes import RolloutFailure
from nemo_gym.rollout_recovery import RunManifest, manifest_path_for
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_rollout_collection import FakeResponse, failing_row, install_fake_server_client
from tests.unit_tests.test_rollout_recovery import runner_config  # noqa: F401
from tests.unit_tests.test_rollout_store import prepared_run, snapshot  # noqa: F401


async def test_native_materialized_judge_input_can_append_reverification(tmp_path, monkeypatch):
    from nemo_gym.rollout_store import RolloutStore

    source = tmp_path / "tasks.jsonl"
    row = {
        "task_id": {"taskset": "native", "task_id": "question-1"},
        "task_input": {
            "responses_create_params": {"input": "What is 6 * 7?"},
            "task_data": {"expected_answer": "42"},
        },
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "_ng_environment_server": "environment",
    }
    source.write_text(json.dumps(row) + "\n")
    servers = {
        "environment": {"environment_servers": {"single_agent_turn": {"resources_server": {"name": "judge"}}}},
        "judge": {"resources_servers": {"example": {}}},
    }
    output = tmp_path / "out.jsonl"
    unscored = row | {"_ng_task_index": 1, "task_id": {"taskset": "native", "task_id": "question-2"}}
    rows = [row, unscored]
    manifest = RunManifest.create(source, rows, {}, servers)
    response = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "42"}]}]}
    # An already classified judge failure exercises the input-shape bridge. Native
    # EpisodeFailure -> judge_failed conversion is a separate producer contract.
    with RolloutStore.start_or_resume(output, lambda: (rows, manifest), resume=False) as store:
        dispatched = store.pending(3)[0]
        store.record_dispatch(dispatched)
        store.record_outcome(
            dispatched
            | {
                "_ng_failure_class": "judge_failed",
                "response": response,
                "_ng_result_type": "single_agent_turn",
                "_ng_task_id": row["task_id"],
            }
        )

        unscored_dispatch = store.pending(3)[1]
        store.record_dispatch(unscored_dispatch)
        store.record_outcome(unscored_dispatch | {"artifact": "completed without a score"})

    async def post(**kwargs):
        assert kwargs["server_name"] == "judge" and kwargs["url_path"] == "/verify"
        payload = kwargs["json"]
        assert "task_input" not in payload and "task_id" not in payload
        assert payload["expected_answer"] == "42"
        assert payload["responses_create_params"] == row["task_input"]["responses_create_params"]
        assert payload["response"] == response and payload["_ng_attempt_index"] == 1
        assert RolloutStore.read(output).coverage()["unknown"] == 1
        return FakeResponse(200, {"reward": 1.0, "response": payload["response"]})

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    client.global_config_dict = OmegaConf.create(servers)
    monkeypatch.setattr(reverification, "setup_server_client", lambda: client)
    monkeypatch.setattr(reverification, "raise_for_status", collection.raise_for_status)
    monkeypatch.setattr(reverification, "get_response_json", collection.get_response_json)
    exported = []
    monkeypatch.setattr(reverification, "get_exporters", lambda: True)
    monkeypatch.setattr(reverification, "export_metrics", lambda metrics: exported.append(metrics))
    monkeypatch.setattr(reverification, "export_rollouts", lambda rows: None)
    config = reverification.RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(output.with_name("out_materialized_inputs.jsonl")),
        rollouts_jsonl_fpath=str(output),
        output_jsonl_fpath=str(output),
        judge_failed_only=True,
        append=True,
        disable_aggregation=True,
    )
    results = await reverification.RolloutReverificationHelper().run_from_config(config)
    assert len(results) == 2 and results[0]["reward"] == 1.0
    assert results[1]["artifact"] == "completed without a score"
    assert exported[-1] == {"coverage/expected": 2, "coverage/scored": 1, "coverage/missing": 1}
    assert results[0]["_ng_task_id"] == row["task_id"] and results[0]["_ng_result_type"] == "single_agent_turn"
    assert RolloutStore.read(output).coverage()["complete"]
    await reverification.RolloutReverificationHelper().run_from_config(config)
    assert client.post.await_count == 1  # No generation or duplicate judge request.


@pytest.mark.parametrize("route_failures", [False, True])
@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize("answer", ["42", ""])
async def test_collected_judge_failure_can_be_reverified_without_inference(
    runner_config, monkeypatch, route_failures, append, answer
):
    runner_config.route_failures_to_sidecar = route_failures
    generated_response = {"output": [{"type": "message", "content": [{"type": "output_text", "text": answer}]}]}

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
                    "failure_kind": "judge_failed",
                    "failure_reason": "Judge unavailable",
                    "mask_sample": True,
                    "instance_config": {"mask_sample": True},
                    "reward": 0.0,
                    "response": generated_response,
                    "ng_trajectory": {
                        "task_id": "1",
                        "rollout_id": "1-0",
                        "turns": [
                            {
                                "invocation_id": "agent",
                                "task_id": "1",
                                "rollout_id": "1-0",
                                "turn_no": 1,
                                "timestamp": 1.0,
                                "step_count": 1,
                                "answer": answer,
                            }
                        ],
                    },
                },
            )
        if row["task"] == 2:
            return FakeResponse(
                200,
                RolloutFailure(
                    episode_id=EpisodeId(rollout_id=logical_rollout_id(row), attempt=0),
                    run_id=row["_ng_run_id"],
                    source="environment",
                    delivery="delivered",
                    failure=EpisodeFailure(
                        failure_reason="No generation saved",
                        terminal=False,
                        failure_kind="judge_failed",
                        stage="verification",
                    ),
                ).model_dump(),
            )
        return FakeResponse(200, {"reward": 0.0, "response": {"output": []}})

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    output = Path(runner_config.output_jsonl_fpath)
    successes = list(read_records(output))
    failures = {row["_ng_task_index"]: row for row in read_records(collection.failures_path_for(output))}
    assert failures[1]["response"] == generated_response
    assert failures[1]["mask_sample"] is True and failures[1]["instance_config"]["mask_sample"] is True
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
    assert not by_task[1].get("mask_sample") and "failure_kind" not in by_task[1]
    verify_request = next(
        call.kwargs["json"] for call in client.post.await_args_list if call.kwargs["url_path"] == "/verify"
    )
    assert "ng_trajectory" not in verify_request
    assert "mask_sample" not in verify_request and "instance_config" not in verify_request
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

    assert by_task[1]["ng_trajectory"] == failures[1]["ng_trajectory"]


async def test_reverify_append_normalizes_repeated_judge_failures(runner_config, monkeypatch):
    response = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "saved answer"}]}]}

    async def post(**kwargs):
        row = kwargs["json"]
        if kwargs["url_path"] == "/verify" or row["_ng_task_index"] == 1:
            return FakeResponse(
                200,
                {
                    "_ng_failure_class": "judge_failed",
                    "_ng_failure_judge_error": "Judge unavailable",
                    "reward": 0.0,
                    "response": response,
                    "grading_notes": "Inspect judge-service.log",
                },
            )
        return FakeResponse(200, {"reward": 0.0, "response": {}})

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    monkeypatch.setattr(reverification, "setup_server_client", lambda: client)
    monkeypatch.setattr(reverification, "_build_agent_to_resources_server_mapping", lambda _: {"my_agent": "rs"})
    monkeypatch.setattr(reverification, "raise_for_status", collection.raise_for_status)
    monkeypatch.setattr(reverification, "get_response_json", collection.get_response_json)
    monkeypatch.setattr(reverification, "get_exporters", list)
    config = reverification.RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(runner_config.materialized_jsonl_fpath),
        rollouts_jsonl_fpath=runner_config.output_jsonl_fpath,
        output_jsonl_fpath=runner_config.output_jsonl_fpath,
        judge_failed_only=True,
        append=True,
        disable_aggregation=True,
    )
    await reverification.RolloutReverificationHelper().run_from_config(config)
    failures = list(read_records(collection.failures_path_for(Path(config.output_jsonl_fpath))))
    assert len(failures) == 2
    recovered = failures[-1]
    failure = RolloutFailure.model_validate(recovered["_ng_failure_record"])
    assert failure.failure.failure_kind == "judge_failed" and failure.failure.stage == "verification"
    assert failure.episode_id.attempt == 1
    assert recovered["grading_notes"] == "Inspect judge-service.log"
    assert recovered["response"] == response and "reward" not in recovered
    assert [call.kwargs["url_path"] for call in client.post.await_args_list].count("/run") == 3
    assert [call.kwargs["url_path"] for call in client.post.await_args_list].count("/verify") == 1


@pytest.mark.parametrize("newest", ["saved_answer", "agent_failure", "no_answer"])
@pytest.mark.parametrize("interrupted", [1, 2])
async def test_judge_only_restart_stops_at_the_newest_recorded_outcome(
    runner_config, monkeypatch, newest, interrupted
):
    from nemo_gym.rollout_store import RolloutStore

    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "5")

    async def post(**kwargs):
        row = kwargs["json"]
        assert kwargs["url_path"] == "/verify"  # Judge-only recovery must never request generation.
        assert row["task"] == 0 and row["response"] == {"id": "new-answer"}
        assert row["_ng_attempt_index"] == 2 + interrupted
        return FakeResponse(200, {"reward": 1.0, "response": row["response"]})

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    monkeypatch.setattr(reverification, "setup_server_client", lambda: client)
    monkeypatch.setattr(reverification, "_build_agent_to_resources_server_mapping", lambda _: {"my_agent": "rs"})
    monkeypatch.setattr(reverification, "raise_for_status", collection.raise_for_status)
    monkeypatch.setattr(reverification, "get_response_json", collection.get_response_json)
    monkeypatch.setattr(reverification, "get_exporters", list)
    output = Path(runner_config.output_jsonl_fpath)
    rows = [failing_row(0) | {"task": 0}]
    source = Path(runner_config.input_jsonl_fpath)
    manifest = RunManifest.create(source, rows, {}, client.global_config_dict)
    with RolloutStore.start_or_resume(output, lambda: (rows, manifest), resume=False) as store:
        row = store.pending(5)[0]
        for index in range(2 + interrupted):
            store.record_dispatch(row | {"_ng_attempt_index": index})
        store.record_outcome(row | {"_ng_failure_class": "judge_failed", "response": {"id": "old-answer"}})
        outcome = row | {
            "_ng_attempt_index": 1,
            "_ng_failure_class": "agent_run_error" if newest == "agent_failure" else "judge_failed",
        }
        if newest == "saved_answer":
            outcome["response"] = {"id": "new-answer"}
        store.record_outcome(outcome)
    before = RolloutStore.read(output).coverage()
    config = reverification.RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(runner_config.materialized_jsonl_fpath),
        rollouts_jsonl_fpath=str(output),
        output_jsonl_fpath=str(output),
        judge_failed_only=True,
        append=True,
        disable_aggregation=True,
    )
    helper = reverification.RolloutReverificationHelper()
    if newest == "saved_answer":
        [recovered] = await helper.run_from_config(config)
        assert recovered["response"] == {"id": "new-answer"} and recovered["reward"] == 1.0
        assert client.post.await_count == 1
        assert RolloutStore.read(output).coverage()["successful"] == 1
    else:
        with pytest.warns(UserWarning, match="Skipping judge"):
            assert await helper.run_from_config(config) == []
        client.post.assert_not_awaited()
        assert RolloutStore.read(output).coverage() == before


@pytest.mark.parametrize("max_attempts", [2, 3])
async def test_cancelled_reverify_append_stops_requests_before_closing_journal(
    runner_config, monkeypatch, max_attempts
):
    from nemo_gym.rollout_store import RolloutStore

    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", str(max_attempts))
    started, stopped = asyncio.Event(), asyncio.Event()
    verify_requests = []
    interrupted = True

    async def post(**kwargs):
        row = kwargs["json"]
        if kwargs["url_path"] == "/verify":
            verify_requests.append(row)
            if not interrupted:
                return FakeResponse(200, {"reward": 1.0, "response": row["response"]})
            started.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()
        result = {"reward": 0, "response": {"id": f"answer-{row['task']}"}}
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
    interrupted = False
    results = await reverification.RolloutReverificationHelper().run_from_config(config)
    retried = verify_requests[1:]
    expected_tasks = [2] if max_attempts == 2 else [1, 2]
    assert [row["task"] for row in retried] == expected_tasks
    assert all(row["response"] == {"id": f"answer-{row['task']}"} for row in retried)
    assert all(row["_ng_attempt_index"] == (2 if row["task"] == 1 else 1) for row in retried)
    assert len(results) == 1 + len(expected_tasks)
    coverage = RolloutStore.read(Path(runner_config.output_jsonl_fpath)).coverage()
    assert coverage["unknown"] == int(max_attempts == 2)
    assert [call.kwargs["url_path"] for call in client.post.await_args_list].count("/run") == 3
    assert await reverification.RolloutReverificationHelper().run_from_config(config) == results
    assert len(verify_requests) == 1 + len(expected_tasks)


@pytest.mark.parametrize("explicit_attempt", [False, True])
async def test_reverify_imported_legacy_attempt_uses_latest_saved_answer(runner_config, monkeypatch, explicit_attempt):
    from nemo_gym.rollout_journal import materialized_path_for
    from nemo_gym.rollout_store import RolloutStore

    output = Path(runner_config.output_jsonl_fpath)
    row = failing_row(0) | {"task": 0}
    materialized_path_for(output).write_text(json.dumps(row) + "\n")
    output.write_text("")
    old = row | {"_ng_failure_class": "judge_failed", "response": {"id": "old", "output": []}}
    latest = row | {"_ng_failure_class": "judge_failed", "response": {"id": "latest", "output": []}}
    if explicit_attempt:
        old["_ng_attempt_index"] = latest["_ng_attempt_index"] = 0
    collection.failures_path_for(output).write_text(json.dumps(old) + "\n" + json.dumps(latest) + "\n")
    with pytest.warns(UserWarning, match="allow_unsafe_resume"):
        store = RolloutStore.start_or_resume(output, lambda: None, resume=True, allow_unsafe=True)
    with store:
        pass
    verified = []

    async def post(**kwargs):
        assert kwargs["url_path"] == "/verify"
        verified.append(kwargs["json"])
        return FakeResponse(200, {"reward": 1.0, "response": kwargs["json"]["response"]})

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    monkeypatch.setattr(reverification, "setup_server_client", lambda: client)
    monkeypatch.setattr(reverification, "_build_agent_to_resources_server_mapping", lambda _: {"my_agent": "rs"})
    monkeypatch.setattr(reverification, "raise_for_status", collection.raise_for_status)
    monkeypatch.setattr(reverification, "get_response_json", collection.get_response_json)
    monkeypatch.setattr(reverification, "get_exporters", list)
    config = reverification.RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(materialized_path_for(output)),
        rollouts_jsonl_fpath=str(output),
        output_jsonl_fpath=str(output),
        judge_failed_only=True,
        append=True,
        disable_aggregation=True,
    )
    results = await reverification.RolloutReverificationHelper().run_from_config(config)
    assert len(verified) == 1
    assert verified[0]["response"] == latest["response"]
    assert verified[0]["_ng_attempt_index"] == 2
    assert results[0]["reward"] == 1.0
    assert await reverification.RolloutReverificationHelper().run_from_config(config) == results
    assert len(verified) == 1


@pytest.mark.parametrize("loose_sidecar", [False, True])
def test_legacy_reverify_read_does_not_infer_strict_history_from_inventory(tmp_path, loose_sidecar):
    from nemo_gym.rollout_journal import materialized_path_for

    output = tmp_path / "legacy.jsonl"
    row = failing_row(0) | {"reward": 1.0}
    output.write_text(json.dumps(row) + "\n")
    inventory = failing_row(0 if loose_sidecar else 1)
    materialized_path_for(output).write_text(json.dumps(inventory) + "\n")
    if loose_sidecar:
        collection.failures_path_for(output).write_text('{"_ng_failure_class":"judge_failed"}\n')
    assert reverification._load_reverified_results(output)[0] == [row]
    manifest_path_for(output).write_text("{}")
    with pytest.raises(ValidationError):
        reverification._load_reverified_results(output)


@pytest.mark.parametrize(
    "latest", ["unknown", "terminal", "omitted", "success", "agent_failure", "no_answer", "exhausted"]
)
def test_judge_retry_inputs_use_history_without_replacing_latest_status(prepared_run, latest):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(5)[0]
        attempts = [row | {"_ng_attempt_index": i} for i in range(4)]
        for attempt in attempts:
            store.record_dispatch(attempt)
        for index in [2, 0, 1]:  # Arrival order differs from attempt order.
            store.record_outcome(
                attempts[index] | {"_ng_failure_class": "judge_failed", "response": {"id": f"answer-{index}"}}
            )
        if latest == "omitted":
            store.record_omission(attempts[3], "Intentionally skipped")
        elif latest == "success":
            store.record_outcome(attempts[3] | {"response": {"id": "completed"}, "reward": 1.0})
        elif latest in {"terminal", "agent_failure", "no_answer"}:
            store.record_outcome(
                attempts[3]
                | {
                    "_ng_failure_class": "agent_run_error" if latest == "agent_failure" else "judge_failed",
                    "_ng_failure_terminal": latest == "terminal",
                }
            )
    reader = RolloutStore.read(output)
    before = snapshot(output)
    coverage = reader.coverage()
    failures = reader.failures()
    payloads = reader.reverification_failures(4 if latest == "exhausted" else 5)
    if latest == "unknown":
        assert len(payloads) == 1 and payloads[0]["response"] == {"id": "answer-2"}
        assert payloads[0]["_ng_attempt_index"] == 2
        [allocated] = reader.for_reverification([row | {"response": payloads[0]["response"]}])
        assert allocated["_ng_attempt_index"] == 4
    elif latest == "no_answer":
        assert len(payloads) == 1 and "response" not in payloads[0]
        assert payloads[0]["_ng_attempt_index"] == 3  # Do not substitute an older answer for a known latest failure.
    else:
        assert payloads == []
    assert reader.coverage() == coverage and reader.failures() == failures
    assert snapshot(output) == before


@pytest.mark.parametrize("known", ["agent_failure", "no_answer", "terminal", "omitted", "success", "none"])
@pytest.mark.parametrize("interrupted", [1, 2])
def test_interruption_does_not_revive_an_older_judge_answer(prepared_run, known, interrupted):
    from nemo_gym.rollout_reverification import _yield_inputs_and_rollouts_paired

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = store.pending(5)[0]
        for index in range(2 + interrupted):
            store.record_dispatch(row | {"_ng_attempt_index": index})
        if known != "none":
            store.record_outcome(row | {"_ng_failure_class": "judge_failed", "response": {"id": "old-answer"}})
            newer = row | {"_ng_attempt_index": 1}
            if known == "omitted":
                store.record_omission(newer, "Do not reuse the earlier answer")
            elif known == "success":
                store.record_outcome(newer | {"reward": 1.0, "response": {"id": "completed"}})
            else:
                store.record_outcome(
                    newer
                    | {
                        "_ng_failure_class": "agent_run_error" if known == "agent_failure" else "judge_failed",
                        "_ng_failure_terminal": known == "terminal",
                    }
                )
    reader = RolloutStore.read(output)
    before = snapshot(output)
    coverage = reader.coverage()
    with pytest.warns(UserWarning, match="Skipping judge"):
        candidates = reader.reverification_failures(5)
        pairs = list(
            _yield_inputs_and_rollouts_paired(materialized_path_for(output), output, selected_rollouts=candidates)
        )
    assert pairs == []
    assert reader.disposition(row) == "unknown"
    assert reader.coverage() == coverage
    assert snapshot(output) == before


def test_reverification_rejects_changed_inputs_before_dispatch(prepared_run):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False):
        pass
    writer = RolloutStore.append_existing(output)
    before = snapshot(output)
    rows, _ = prepare()
    for invalid in (rows[0] | {"_ng_task_index": 999}, rows[0] | {"agent_ref": {"name": "other"}}):
        with pytest.raises(ConfigError, match="Reverification input"):
            writer.for_reverification([invalid | {"response": {}}])
    assert snapshot(output) == before
    journal_path_for(output).unlink()
    with pytest.raises(ConfigError, match="without attempt history"):
        RolloutStore.append_existing(output)


@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("tail", [None, "output", "failures"])
async def test_journal_invalid_judge_opt_in_reuses_saved_answer(runner_config, monkeypatch, append, alias, tail):
    async def post(**kwargs):
        row = kwargs["json"]
        if kwargs["url_path"] == "/verify":
            return FakeResponse(200, {"reward": 1.0, "response": row["response"]})
        return FakeResponse(
            200,
            {"reward": 0.0, "response": {"output": [str(row["task"])]}, "invalid_judge_response": row["task"] == 1},
        )

    client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    await RolloutCollectionHelper().run_from_config(runner_config)
    source = Path(runner_config.output_jsonl_fpath)
    answer = RolloutStore.read(source).selected("success")[1]["response"]
    if tail:
        target = source if tail == "output" else collection.failures_path_for(source)
        with target.open("ab") as handle:
            handle.write(b'{"interrupted":')
    if alias:
        shortcut = source.with_name("shortcut.jsonl")
        shortcut.symlink_to(source)
        source = shortcut
    monkeypatch.setattr(reverification, "setup_server_client", lambda: client)
    monkeypatch.setattr(reverification, "_build_agent_to_resources_server_mapping", lambda _: {"my_agent": "rs"})
    monkeypatch.setattr(reverification, "raise_for_status", collection.raise_for_status)
    monkeypatch.setattr(reverification, "get_response_json", collection.get_response_json)
    monkeypatch.setattr(reverification, "get_exporters", list)
    config = reverification.RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(runner_config.materialized_jsonl_fpath),
        rollouts_jsonl_fpath=str(source),
        output_jsonl_fpath=str(source if append else source.with_name("judged.jsonl")),
        judge_failed_only=True,
        append=append,
        retry_invalid_judge_responses=True,
        disable_aggregation=True,
    )
    result = await reverification.RolloutReverificationHelper().run_from_config(config)
    assert len(result) == 3
    assert all(row["_ng_environment_server"] == "my_environment_server" for row in result)
    assert result[1]["response"] == answer and result[1]["reward"] == 1.0
    assert [c.kwargs["url_path"] for c in client.post.await_args_list].count("/run") == 3
    assert [c.kwargs["url_path"] for c in client.post.await_args_list].count("/verify") == 1
    if append:
        assert RolloutStore.read(source).coverage()["attempts"] == 4


def test_projected_judge_recovery_does_not_seed_superseded_success(prepared_run):
    source, prepare = prepared_run
    with RolloutStore.start_or_resume(source, prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(row | {"reward": 1.0, "response": {"id": "stale"}})
        store.record_dispatch(row | {"_ng_attempt_index": 1})
    output = source.with_name("projection.jsonl")
    assert reverification._seed_output_with_successes(source, output) == set()
    assert output.read_bytes() == b""
    manifest_path_for(source).unlink()
    with pytest.raises(ConfigError, match="without a run manifest"):
        RolloutStore.append_existing(source)
