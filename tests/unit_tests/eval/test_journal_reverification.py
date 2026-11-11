# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-score selected saved answers without changing their source run's recovery history."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest
from omegaconf import OmegaConf

import nemo_gym.eval.rollout_reverification as reverify
from nemo_gym.config.config_types import ConfigError
from nemo_gym.rollouts.rollout_records import RUN_ID_KEY, RolloutRecord, materialized_path_for, read_records
from nemo_gym.rollouts.rollout_recovery import RunManifest, config_identity_path_for, manifest_path_for, run_lock
from nemo_gym.rollouts.rollout_store import RolloutStore
from nemo_gym.utils.path_utils import aggregation_sources_path_for
from tests.helpers.single_agent_turn import _agent_response, _environment_server, _request, _Response
from tests.unit_tests.rollouts.test_rollout_store import prepared_run, snapshot  # noqa: F401


def _config(output: Path, destination: Path, **kwargs) -> reverify.RolloutReverificationConfig:
    return reverify.RolloutReverificationConfig(
        materialized_inputs_jsonl_fpath=str(materialized_path_for(output)),
        rollouts_jsonl_fpath=str(output),
        output_jsonl_fpath=str(destination),
        disable_aggregation=True,
        upload_rollouts=False,
        **kwargs,
    )


def _replay_servers() -> dict:
    return {
        "agent": {"responses_api_agents": {"fixture": {"resources_server": {"name": "resources"}}}},
        "resources": {"resources_servers": {"fixture": {}}},
        **{
            name: {
                "environment_servers": {
                    "single_agent_turn": {
                        "entrypoint": entrypoint,
                        "agent_server": {"name": "agent", "type": "responses_api_agents"},
                        "resources_server": {"name": "resources", "type": "resources_servers"},
                    }
                }
            }
            for name, entrypoint in (("plain", "app.py"), ("special", "opencode.py"))
        },
    }


def _saved_run(tmp_path: Path, outcomes: list[tuple[str, dict]]) -> tuple[Path, list[dict]]:
    rows = [
        {
            "_ng_task_index": task,
            "_ng_rollout_index": 0,
            "_ng_environment_server": route,
            "agent_ref": {"name": "agent"},
            "responses_create_params": {"input": []},
        }
        for task, (route, _) in enumerate(outcomes)
    ]
    source = tmp_path / "inputs.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    output = tmp_path / "rollouts.jsonl"
    saved = []
    with RolloutStore.start_or_resume(
        output, lambda: (rows, RunManifest.create(source, rows, {}, _replay_servers())), resume=False
    ) as store:
        for row, (_, result) in zip(store.pending(), outcomes, strict=True):
            store.record_dispatch(row)
            saved.append(row | result)
            store.record_outcome(saved[-1])
    return output, saved


def _replay_client(monkeypatch, *, servers: dict | None = None):
    async def post(*, server_name, url_path, json, **kwargs):
        assert (server_name, url_path) == ("resources", "/verify")
        return _Response(json | {"reward": 0.25, "evaluation_completed": True})

    client = SimpleNamespace(
        global_config_dict=OmegaConf.create(servers or _replay_servers()),
        get=AsyncMock(return_value=_Response("stateless")),
        post=AsyncMock(side_effect=post),
    )
    monkeypatch.setattr(reverify, "setup_server_client", lambda: client)
    monkeypatch.setattr(reverify, "get_exporters", list)
    return client


async def test_plain_native_response_replays_from_its_original_environment(tmp_path, monkeypatch):
    environment, producer = _environment_server()
    response = _agent_response().model_dump(mode="json")
    response["opencode_evidence"] = {"turns_used": 1, "finished_naturally": True, "tool_reward": None}
    producer.responses = [_Response({"responses_create_params": {"input": []}, "response": response, "reward": 1})]
    from environment_servers.single_agent_turn.app import SessionHandles

    result = await environment._verify(
        _request(),
        SessionHandles(resources_session_id="resource", agent_session_id="agent"),
        type(_agent_response()).model_validate(response),
        None,
    )
    assert len(producer.calls) == 1 and producer.calls[0][1] == "/verify"
    assert producer.calls[0][2]["json"]["response"] == response
    output, _ = _saved_run(tmp_path, [("plain", result.model_dump(mode="json"))])
    current = _replay_servers()
    current["plain"]["environment_servers"]["single_agent_turn"]["entrypoint"] = "opencode.py"
    client = _replay_client(monkeypatch, servers=current)
    before = snapshot(output)
    results = await reverify.RolloutReverificationHelper().run_from_config(
        _config(output, tmp_path / "rescored.jsonl")
    )
    assert [row["reward"] for row in results] == [0.25]
    client.post.assert_awaited_once()
    assert client.post.await_args.kwargs["json"]["response"] == response
    assert list(read_records(tmp_path / "rescored.jsonl")) == results
    assert {name: data for name, data in snapshot(output).items() if name in before} == before


@pytest.mark.parametrize("kind", ["native", "failed", "verification_error"])
@pytest.mark.parametrize("resume", [False, True])
async def test_original_special_environment_refuses_before_output_changes(tmp_path, monkeypatch, kind, resume):
    response = _agent_response().model_dump(mode="json")
    response["opencode_evidence"] = {"turns_used": 1, "finished_naturally": True, "tool_reward": None}
    result = {"response": response, "reward": 1.0}
    if kind == "failed":
        result = {
            "response": response | {"status": "failed", "error": {"code": "server_error", "message": "limit"}},
            "reward": 0.0,
            "scorer": "agent-failure",
        }
        result["response"].pop("opencode_evidence")
    elif kind == "verification_error":
        response["opencode_evidence"]["verification_error"] = "unusable native evidence"
        result = {"_ng_failure_class": "judge_failed", "_ng_failure_partial_response": response}
    output, [saved] = _saved_run(tmp_path, [("special", result)])
    destination = tmp_path / "rescored.jsonl"
    config = _config(
        output, destination, overwrite=True, resume_from_cache=resume, judge_failed_only=kind == "verification_error"
    )
    if resume and kind != "verification_error":
        store = RolloutStore.read(output)
        cached = reverify._project_reverification_row(saved) | {
            "_ng_reverification_source_answer": reverify._rescoring_source_answer(saved, store.manifest.run_id)
        }
        destination.write_bytes(orjson.dumps(cached) + b"\n")
    else:
        destination.write_bytes(orjson.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0}) + b"\n")
    aggregation_sources_path_for(destination).write_text("keep the previous shard list\n")
    current = _replay_servers()
    current["special"]["environment_servers"]["single_agent_turn"]["entrypoint"] = "app.py"
    client = _replay_client(monkeypatch, servers=current)
    before = snapshot(output)
    with pytest.raises(ConfigError, match="OpenCode environment grading"):
        await reverify.RolloutReverificationHelper().run_from_config(config)
    client.post.assert_not_awaited()
    assert snapshot(output) == before


async def test_unverified_identity_cannot_authorize_native_replay(tmp_path, monkeypatch):
    response = _agent_response().model_dump(mode="json") | {"opencode_evidence": {"tool_reward": None}}
    output, _ = _saved_run(tmp_path, [("plain", {"response": response, "reward": 1.0})])
    identity = config_identity_path_for(manifest_path_for(output))
    identity.write_bytes(identity.read_bytes().replace(b'"app.py"', b'"other.py"'))
    client = _replay_client(monkeypatch)
    before = snapshot(output)
    with pytest.raises(ConfigError, match="digest-verified original plain"):
        await reverify.RolloutReverificationHelper().run_from_config(_config(output, tmp_path / "rescored.jsonl"))
    client.post.assert_not_awaited()
    assert snapshot(output) == before


async def test_special_source_preflight_precedes_withdrawal_and_seeding(tmp_path, monkeypatch):
    response = _agent_response().model_dump(mode="json")
    output, [failure] = _saved_run(
        tmp_path, [("special", {"response": response, "_ng_failure_class": "judge_failed", "_ng_failure_id": "judge"})]
    )
    with run_lock(output), RolloutStore.open_for_rejudgment(output) as store:
        pending = store.admit_rejudgment(failure)
    client = _replay_client(monkeypatch)
    before = snapshot(output)
    with pytest.raises(ConfigError, match="OpenCode environment grading"):
        await reverify.RolloutReverificationHelper().run_from_config(
            _config(
                output,
                tmp_path / "rescored.jsonl",
                judge_failed_only=True,
                settle_into_source=True,
                withdraw_unresolved_admissions=True,
            )
        )
    client.post.assert_not_awaited()
    assert snapshot(output) == before
    assert RolloutStore.read(output).manifest.pending_rejudge["0-0"].operation_id == pending.operation_id


async def test_atif_uses_original_materialized_source_before_projection(tmp_path, monkeypatch):
    output, _ = _saved_run(
        tmp_path, [("special", {"response": _agent_response().model_dump(mode="json"), "reward": 0})]
    )
    manifest = tmp_path / "external.jsonl"
    manifest.write_bytes(
        orjson.dumps({"trajectory_path": "not-read.json", "_ng_task_index": 0, "_ng_rollout_index": 0}) + b"\n"
    )
    current = _replay_servers()
    current["special"]["environment_servers"]["single_agent_turn"]["entrypoint"] = "app.py"
    client = _replay_client(monkeypatch, servers=current)
    before = snapshot(output)
    with pytest.raises(ConfigError, match="OpenCode environment grading"):
        await reverify.RolloutReverificationHelper().run_from_config(
            reverify.RolloutReverificationConfig(
                input_format="atif",
                materialized_inputs_jsonl_fpath=str(materialized_path_for(output)),
                atif_manifest_jsonl_fpath=str(manifest),
                output_jsonl_fpath=str(tmp_path / "rescored.jsonl"),
                disable_aggregation=True,
                upload_rollouts=False,
            )
        )
    client.post.assert_not_awaited()
    assert snapshot(output) == before


async def test_unselected_special_history_does_not_block_plain_saved_answer(tmp_path, monkeypatch):
    response = _agent_response().model_dump(mode="json")
    output, saved = _saved_run(
        tmp_path, [("special", {"response": response, "reward": 0}), ("plain", {"response": response, "reward": 1})]
    )
    with RolloutStore.open_for_rejudgment(output) as store:
        newer = dict(saved[0])
        store.allocate_attempt(newer)
        store.record_outcome(newer | {"_ng_failure_class": "agent_run_error"})
    client = _replay_client(monkeypatch)
    before = snapshot(output)
    result = await reverify.RolloutReverificationHelper().run_from_config(_config(output, tmp_path / "rescored.jsonl"))
    assert [row["_ng_task_index"] for row in result] == [1]
    assert client.post.await_args.kwargs["json"]["_ng_task_index"] == 1
    assert {name: data for name, data in snapshot(output).items() if name in before} == before


async def test_recovery_carries_special_agent_failure_zero_without_regrading_it(tmp_path, monkeypatch):
    response = _agent_response().model_dump(mode="json")
    output, _ = _saved_run(
        tmp_path,
        [
            (
                "special",
                {
                    "response": response | {"status": "failed", "error": {"code": "server_error", "message": "limit"}},
                    "reward": 0,
                    "scorer": "agent-failure",
                },
            ),
            ("plain", {"response": response, "_ng_failure_class": "judge_failed", "_ng_failure_id": "judge"}),
        ],
    )
    client = _replay_client(monkeypatch)
    before = snapshot(output)
    result = await reverify.RolloutReverificationHelper().run_from_config(
        _config(output, tmp_path / "rescored.jsonl", judge_failed_only=True)
    )
    assert [(row["_ng_task_index"], row["reward"]) for row in result] == [(0, 0), (1, 0.25)]
    assert result[0]["scorer"] == "agent-failure"
    client.post.assert_awaited_once()
    assert client.post.await_args.kwargs["json"]["_ng_task_index"] == 1
    assert {name: data for name, data in snapshot(output).items() if name in before} == before


async def test_special_saved_result_is_checked_before_settlement(tmp_path, monkeypatch):
    from tests.unit_tests.eval.test_reverify_saved_results import verify_transport_masked

    output, _ = _saved_run(tmp_path, [("special", verify_transport_masked())])
    with run_lock(output):
        pass
    client = _replay_client(monkeypatch)
    before = snapshot(output)
    with pytest.raises(ConfigError, match="OpenCode environment grading"):
        await reverify.RolloutReverificationHelper().run_from_config(
            _config(output, tmp_path / "rescored.jsonl", judge_failed_only=True, settle_into_source=True)
        )
    client.post.assert_not_awaited()
    assert snapshot(output) == before
    assert RolloutStore.read(output).manifest.pending_rejudge == {}


@pytest.mark.parametrize("newest", ["success", "failure", "interrupted"])
@pytest.mark.parametrize("alias", [False, True])
async def test_rescore_uses_only_selected_completed_answers(prepared_run, monkeypatch, newest, alias):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        first, second = store.pending()
        store.record_dispatches([first, second])
        store.record_outcome(first | {"reward": 0.0, "response": {"answer": "old"}})
        store.record_outcome(second | {"reward": 0.0, "response": {"answer": "other"}})
        newer = dict(first)
        store.allocate_attempt(newer)
        if newest == "success":
            store.record_outcome(newer | {"reward": 0.0, "response": {"answer": "new"}})
        elif newest == "failure":
            store.record_outcome(newer | {"_ng_failure_class": "agent_run_error"})
        run_id = store.manifest.run_id
    source = output
    if alias:
        source = output.with_name("shortcut.jsonl")
        source.symlink_to(output)
    before = snapshot(output)
    seen = []
    monkeypatch.setattr(reverify, "_guard_reverify_mode", AsyncMock(return_value=None))
    monkeypatch.setattr(reverify, "get_exporters", list)

    async def judge(row):
        seen.append(row)
        return row, {"response": row["response"], "reward": 0.75}

    monkeypatch.setattr(reverify, "_run_verification_payloads", lambda payloads, **_: [judge(r) for r in payloads])
    destination = output.parent / "rescored" / "rollouts.jsonl"
    config = _config(source, destination)
    config.materialized_inputs_jsonl_fpath = str(materialized_path_for(output))
    result = await reverify.RolloutReverificationHelper().run_from_config(config)
    expected = {"new", "other"} if newest == "success" else {"other"}
    assert {row["response"]["answer"] for row in seen} == expected
    assert len(result) == len(expected) and all(row["reward"] == 0.75 for row in result)
    assert all(row["_ng_reverification_source_run_id"] == run_id and RUN_ID_KEY not in row for row in result)
    assert snapshot(output) == before
    assert RolloutStore.read(destination, import_legacy=False) is None
    seen.clear()
    await reverify.RolloutReverificationHelper().run_from_config(config.model_copy(update={"resume_from_cache": True}))
    assert seen == []
    assert list(read_records(destination)) == result
    assert snapshot(output) == before


async def test_rescore_accepts_a_judge_failure_settled_by_the_stack(tmp_path, monkeypatch):
    from tests.unit_tests.eval.test_rejudge_saved_judge_failures import _run, _settle

    output, failures = _run(tmp_path, judge_failures=1)
    _settle(output, failures[0], 0.25)
    before = snapshot(output)
    monkeypatch.setattr(reverify, "_guard_reverify_mode", AsyncMock(return_value=None))
    monkeypatch.setattr(reverify, "get_exporters", list)
    seen = []

    async def judge(row):
        seen.append(row)
        return row, {"response": row["response"], "reward": 0.5}

    monkeypatch.setattr(reverify, "_run_verification_payloads", lambda payloads, **_: [judge(r) for r in payloads])
    result = await reverify.RolloutReverificationHelper().run_from_config(
        _config(output, tmp_path / "rescored" / "rollouts.jsonl")
    )
    assert len(seen) == len(result) == 3
    assert all(row["reward"] == 0.5 for row in result)
    assert snapshot(output) == before
    assert RolloutStore.read(output).manifest.pending_rejudge == {}


def test_seeded_projection_keeps_selected_successes_and_source_provenance(prepared_run):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        first, second = store.pending()
        store.record_dispatches([first, second])
        store.record_outcome(first | {"reward": 1.0, "response": {"answer": "stale"}})
        store.record_outcome(second | {"reward": 1.0, "response": {"answer": "selected"}})
        newer = dict(first)
        store.allocate_attempt(newer)
        store.record_outcome(newer | {"_ng_failure_class": "judge_failed", "response": {"answer": "pending"}})
        run_id = store.manifest.run_id
    destination = output.parent / "projected" / "rollouts.jsonl"
    destination.parent.mkdir()
    before = snapshot(output)
    assert reverify._seed_output_with_successes(output, destination) == {(1, 0)}
    [saved] = read_records(destination)
    assert saved["response"]["answer"] == "selected"
    assert saved["_ng_reverification_source_run_id"] == run_id and RUN_ID_KEY not in saved
    assert RolloutStore.read(destination, import_legacy=False) is None
    assert reverify._seed_output_with_successes(output, destination) == {(1, 0)}
    assert len(list(read_records(destination))) == 1
    assert snapshot(output) == before


async def test_rescore_source_provenance_stays_out_of_verifier_request(tmp_path, monkeypatch):
    from tests.unit_tests.rollouts.test_rollout_collection import FakeResponse, install_fake_server_client

    reply = FakeResponse(200, {"response": {}, "reward": 1.0})
    client = install_fake_server_client(monkeypatch, AsyncMock(return_value=reply))
    monkeypatch.setattr(reverify, "setup_server_client", lambda: client)
    monkeypatch.setattr(reverify, "_build_agent_to_resources_server_mapping", lambda _: {"agent": "judge"})
    row = {
        "agent_ref": {"name": "agent"},
        "response": {},
        "_ng_reverification_source_run_id": "source-run",
        "_ng_reverification_source_answer": {"run_id": "source-run"},
    }
    await next(reverify._run_verification_payloads([row]))
    assert "_ng_reverification_source_run_id" not in client.post.call_args.kwargs["json"]
    assert "_ng_reverification_source_answer" not in client.post.call_args.kwargs["json"]


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"error": "bad"},
        {"reward": True, "response": {}},
        {"reward": float("inf"), "response": {}},
        {"reward": 1.0, "response": []},
    ],
)
async def test_rescore_rejects_malformed_completion_and_can_resume(prepared_run, monkeypatch, reply):
    from nemo_gym.config.config_types import ConfigError

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for row in store.pending():
            store.record_dispatch(row)
            store.record_outcome(row | {"reward": 0.0, "response": {"answer": "saved"}})
    before = snapshot(output)
    monkeypatch.setattr(reverify, "_guard_reverify_mode", AsyncMock(return_value=None))
    monkeypatch.setattr(reverify, "get_exporters", list)
    destination = output.parent / "rescored" / "rollouts.jsonl"

    async def malformed(row):
        return row, reply

    monkeypatch.setattr(reverify, "_run_verification_payloads", lambda rows, **_: [malformed(r) for r in rows[:1]])
    config = _config(output, destination)
    with pytest.raises(ConfigError, match="response object and a finite numeric reward"):
        await reverify.RolloutReverificationHelper().run_from_config(config)
    assert not destination.read_bytes()

    async def healthy(row):
        return row, {"response": row["response"], "reward": 0.0}

    monkeypatch.setattr(reverify, "_run_verification_payloads", lambda rows, **_: [healthy(r) for r in rows])
    result = await reverify.RolloutReverificationHelper().run_from_config(
        config.model_copy(update={"resume_from_cache": True})
    )
    assert len(result) == 2 and all(row["reward"] == 0.0 for row in result)
    assert snapshot(output) == before


def _save_two_answers(prepared_run):
    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        first, second = store.pending()
        store.record_dispatches([first, second])
        # A masked infrastructure result can receive a new execution during normal collection resume.
        store.record_outcome(
            first
            | {
                "reward": 0.0,
                "response": {"answer": "old"},
                "mask_sample": True,
                "failure_kind": "session_lost",
            }
        )
        store.record_outcome(second | {"reward": 1.0, "response": {"answer": "other"}})
    return output, prepare


def _install_scoring_judge(monkeypatch, status="success"):
    seen = []
    monkeypatch.setattr(reverify, "_guard_reverify_mode", AsyncMock(return_value=None))
    monkeypatch.setattr(reverify, "get_exporters", list)

    async def judge(row):
        seen.append(row["response"]["answer"])
        result = {"response": row["response"], "reward": 0.0}
        if status != "success":
            result.update(_ng_failure_class="judge_failed", _ng_failure_terminal=status == "terminal")
        return row, result

    monkeypatch.setattr(reverify, "_run_verification_payloads", lambda rows, **_: [judge(row) for row in rows])
    return seen


@pytest.mark.parametrize("newest", ["success", "failure", "interrupted"])
@pytest.mark.parametrize("cached_status", ["success", "terminal", "exhausted"])
@pytest.mark.parametrize("alias", [False, True])
async def test_rescore_resume_rejects_a_superseded_cached_answer(
    prepared_run, monkeypatch, newest, cached_status, alias
):
    from nemo_gym.config.config_types import ConfigError

    if cached_status == "exhausted":
        monkeypatch.setattr(reverify, "_get_max_rollout_attempts", lambda: 1)
    output, prepare = _save_two_answers(prepared_run)
    destination = output.parent / "rescored" / "rollouts.jsonl"
    _install_scoring_judge(monkeypatch, cached_status)
    config = _config(output, destination, limit=1)
    await reverify.RolloutReverificationHelper().run_from_config(config)
    with RolloutStore.start_or_resume(output, prepare, resume=True) as store:
        [retry] = store.pending()
        store.record_dispatch(retry)
        assert retry["_ng_attempt_index"] == 1
        if newest == "success":
            store.record_outcome(retry | {"reward": 1.0, "response": {"answer": "new"}})
        elif newest == "failure":
            store.record_outcome(retry | {"_ng_failure_class": "agent_run_error"})
    source = output
    if alias:
        source = output.with_name("shortcut.jsonl")
        source.symlink_to(output)
    before = snapshot(output)
    cached = snapshot(destination)
    seen = _install_scoring_judge(monkeypatch)
    resumed = config.model_copy(update={"rollouts_jsonl_fpath": str(source), "resume_from_cache": True, "limit": None})
    with pytest.raises(ConfigError, match="fresh output path"):
        await reverify.RolloutReverificationHelper().run_from_config(resumed)
    assert seen == [] and snapshot(destination) == cached and snapshot(output) == before
    fresh = await reverify.RolloutReverificationHelper().run_from_config(
        _config(output, output.parent / "fresh" / "rollouts.jsonl")
    )
    assert {row["response"]["answer"] for row in fresh} == ({"new", "other"} if newest == "success" else {"other"})


@pytest.mark.parametrize("alias", [False, True])
async def test_rescore_partial_resume_reuses_zero_reward_without_rejudging(prepared_run, monkeypatch, alias):
    output, _ = _save_two_answers(prepared_run)
    destination = output.parent / "rescored" / "rollouts.jsonl"
    seen = _install_scoring_judge(monkeypatch)
    config = _config(output, destination, limit=1)
    [first] = await reverify.RolloutReverificationHelper().run_from_config(config)
    assert first["reward"] == 0.0 and seen == ["old"]
    source = output
    if alias:
        source = output.with_name("shortcut.jsonl")
        source.symlink_to(output)
    before = snapshot(output)
    seen.clear()
    results = await reverify.RolloutReverificationHelper().run_from_config(
        config.model_copy(update={"rollouts_jsonl_fpath": str(source), "resume_from_cache": True, "limit": None})
    )
    assert seen == ["other"] and len(results) == 2
    assert list(read_records(destination))[0] == first
    assert snapshot(output) == before


@pytest.mark.parametrize("change", ["missing_binding", "wrong_answer", "wrong_attempt", "wrong_run"])
async def test_rescore_resume_refuses_unbound_or_foreign_cache(prepared_run, monkeypatch, change):
    from nemo_gym.config.config_types import ConfigError

    output, _ = _save_two_answers(prepared_run)
    destination = output.parent / "rescored" / "rollouts.jsonl"
    _install_scoring_judge(monkeypatch)
    config = _config(output, destination, limit=1)
    await reverify.RolloutReverificationHelper().run_from_config(config)
    [cached] = read_records(destination)
    if change == "missing_binding":
        cached.pop("_ng_reverification_source_answer")
    else:
        key, value = {
            "wrong_answer": ("answer_sha256", "0" * 64),
            "wrong_attempt": ("attempt", 42),
            "wrong_run": ("run_id", "another-run"),
        }[change]
        cached["_ng_reverification_source_answer"][key] = value
    destination.write_bytes(orjson.dumps(cached) + b"\n")
    before = snapshot(destination)
    seen = _install_scoring_judge(monkeypatch)
    with pytest.raises(ConfigError, match="fresh output path"):
        await reverify.RolloutReverificationHelper().run_from_config(
            config.model_copy(update={"resume_from_cache": True})
        )
    assert seen == [] and snapshot(destination) == before


@pytest.mark.parametrize("limit", [1, None])
async def test_rescore_decodes_only_requested_selected_answers(tmp_path, monkeypatch, limit):
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(6)]
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))

    def prepare():
        return rows, RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {"impl": {}}}})

    output = tmp_path / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for row in store.pending():
            store.record_dispatch(row)
            store.record_outcome(
                row | {"reward": 0.0, "response": {"answer": str(row["_ng_task_index"]), "padding": "X" * 262144}}
            )
    original_read = RolloutRecord._read
    decoded = []

    def counted_read(record, file):
        result = original_read(record, file)
        if record.path == output:
            decoded.append(result["_ng_task_index"])
        return result

    monkeypatch.setattr(RolloutRecord, "_read", counted_read)
    seen = _install_scoring_judge(monkeypatch)
    before = snapshot(output)
    results = await reverify.RolloutReverificationHelper().run_from_config(
        _config(output, tmp_path / "rescored" / "rollouts.jsonl", limit=limit)
    )
    expected_indices = list(range(6 if limit is None else limit))
    assert decoded == expected_indices
    expected = [str(i) for i in expected_indices]
    assert seen == expected and len(results) == len(expected)
    assert snapshot(output) == before


@pytest.mark.parametrize("limit", [1, None])
@pytest.mark.parametrize("alias", [False, True])
async def test_rescore_limit_does_not_interpret_unrequested_completion(prepared_run, monkeypatch, limit, alias):
    from nemo_gym.config.config_types import ConfigError

    output, prepare = prepared_run
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        first, second = store.pending()
        store.record_dispatches([first, second])
        store.record_outcome(first | {"response": {}, "reward": 1.0})
        # An arbitrary completion is valid in a saved run, even when it has no answer /verify can grade.
        store.record_outcome(second | {"reward": 1.0})
    source = output
    if alias:
        source = output.with_name("shortcut.jsonl")
        source.symlink_to(output)
    before = snapshot(output)
    seen = []
    monkeypatch.setattr(reverify, "_guard_reverify_mode", AsyncMock(return_value=None))
    monkeypatch.setattr(reverify, "get_exporters", list)

    async def judge(row):
        seen.append(row["_ng_task_index"])
        return row, {"response": row["response"], "reward": 1.0}

    monkeypatch.setattr(reverify, "_run_verification_payloads", lambda rows, **_: [judge(row) for row in rows])
    config = _config(output, output.parent / "rescored" / "rollouts.jsonl", limit=limit)
    config.rollouts_jsonl_fpath = str(source)
    if limit == 1:
        [result] = await reverify.RolloutReverificationHelper().run_from_config(config)
        assert seen == [0] and result["response"] == {} and result["reward"] == 1.0
    else:
        with pytest.raises(ConfigError, match="has no `response`"):
            await reverify.RolloutReverificationHelper().run_from_config(config)
        assert seen == []
    assert snapshot(output) == before


def test_repeat_plan_uses_verified_source_route_for_a_task_without_any_outcome(tmp_path):
    rows = [{"_ng_task_index": task, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for task in range(2)]
    source = tmp_path / "inputs.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    output = tmp_path / "rollouts.jsonl"
    servers = _replay_servers()
    del servers["special"]
    with RolloutStore.start_or_resume(
        output, lambda: (rows, RunManifest.create(source, rows, {}, servers)), resume=False
    ) as store:
        first = store.pending()[0]
        store.record_dispatch(first)
        store.record_outcome(first | {"_ng_environment_server": "plain", "reward": 1})
    plan = reverify._reverification_repeat_inventory(
        [materialized_path_for(output)],
        (output,),
        [],
        {"agent": "resources"},
        {"resources": {"resources_servers": {"fixture": {}}}},
    )
    assert plan == {("plain", "resources"): {0: [0], 1: [0]}}


async def test_verifier_owned_tool_reward_replays_with_original_execution_evidence(tmp_path, monkeypatch):
    evidence = {"schema_version": "2.0", "source": "opencode", "records": []}
    output, _ = _saved_run(
        tmp_path,
        [
            (
                "plain",
                {
                    "response": _agent_response().model_dump(mode="json"),
                    "reward": 1.1,
                    "opencode_tool_reward": {"auxiliary_reward": 0.1},
                    "ng_agent_observations": evidence,
                },
            )
        ],
    )
    client = _replay_client(monkeypatch)
    await reverify.RolloutReverificationHelper().run_from_config(_config(output, tmp_path / "rescored.jsonl"))
    assert client.post.await_args.kwargs["json"]["ng_agent_observations"] == evidence
    assert "opencode_tool_reward" not in client.post.await_args.kwargs["json"]
