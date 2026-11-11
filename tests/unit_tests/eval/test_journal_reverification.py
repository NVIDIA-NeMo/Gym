# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-score selected saved answers without changing their source run's recovery history."""

from pathlib import Path
from unittest.mock import AsyncMock

import orjson
import pytest

import nemo_gym.eval.rollout_reverification as reverify
from nemo_gym.rollouts.rollout_records import RUN_ID_KEY, RolloutRecord, materialized_path_for, read_records
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore
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
            for name, entrypoint in (("plain", "app.py"), ("other", "app.py"))
        },
    }


def test_repeat_plan_uses_verified_source_route_for_a_task_without_any_outcome(tmp_path):
    rows = [{"_ng_task_index": task, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for task in range(2)]
    source = tmp_path / "inputs.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    output = tmp_path / "rollouts.jsonl"
    servers = _replay_servers()
    del servers["other"]
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
