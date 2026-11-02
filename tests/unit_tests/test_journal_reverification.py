# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-score selected saved answers without changing their source run's recovery history."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import nemo_gym.rollout_reverification as reverify
from nemo_gym.rollout_records import RUN_ID_KEY, materialized_path_for, read_records
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_rollout_store import prepared_run, snapshot  # noqa: F401


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
    from tests.unit_tests.test_rejudge_saved_judge_failures import _run, _settle

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
    from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client

    reply = FakeResponse(200, {"response": {}, "reward": 1.0})
    client = install_fake_server_client(monkeypatch, AsyncMock(return_value=reply))
    monkeypatch.setattr(reverify, "setup_server_client", lambda: client)
    monkeypatch.setattr(reverify, "_build_agent_to_resources_server_mapping", lambda _: {"agent": "judge"})
    row = {"agent_ref": {"name": "agent"}, "response": {}, "_ng_reverification_source_run_id": "source-run"}
    await next(reverify._run_verification_payloads([row]))
    assert "_ng_reverification_source_run_id" not in client.post.call_args.kwargs["json"]


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
