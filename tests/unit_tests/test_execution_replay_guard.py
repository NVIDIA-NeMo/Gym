# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollout_recovery import RunManifest
from nemo_gym.rollout_store import RolloutStore


@pytest.mark.parametrize(
    "case,expected_pending",
    [
        ("unowned_route", 0),
        ("known_infra", 1),
        ("no_report_legacy", 1),
        ("dangling_cause", 0),
        ("nonstring_cause", 0),
        ("cycle", 0),
        ("missing_selected", 0),
        ("nonstring_selected", 0),
        ("empty_selected", 0),
        ("malformed_events", 0),
    ],
)
def test_selected_execution_evidence_stops_replay_in_store_and_legacy_cache(
    tmp_path, monkeypatch, case, expected_pending
):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "3")
    identity = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(identity) + "\n")
    events = {
        "dispatch": {
            "failure_kind": "collection_dispatch_failed",
            "boundary": "collection",
            "operation": "collection.run",
            "cause_id": "route",
        },
        "route": {"failure_kind": "agent_route_failed", "boundary": "agent", "operation": "agent.run"},
    }
    failure = {
        "_ng_failure_class": "agent_run_error",
        "_ng_failure_id": "dispatch",
        "_ng_failure_events": events,
        "_ng_elapsed_s": 7.0,
    }
    if case == "known_infra":
        events["route"]["cause_id"] = "cause"
        events["cause"] = {
            "failure_kind": "provider_unavailable",
            "boundary": "harness",
            "operation": "sandbox.create",
        }
    elif case == "no_report_legacy":
        failure = {
            "_ng_failure_class": "agent_run_error",
            "failure_kind": "provider_unavailable",
            "_ng_elapsed_s": 7.0,
        }
    elif case == "dangling_cause":
        events["route"]["cause_id"] = "absent"
    elif case == "nonstring_cause":
        events["route"]["cause_id"] = 7
    elif case == "cycle":
        events["route"]["cause_id"] = "dispatch"
    elif case == "missing_selected":
        failure["_ng_failure_id"] = "absent"
    elif case == "nonstring_selected":
        failure["_ng_failure_id"] = 7
    elif case == "empty_selected":
        failure["_ng_failure_id"] = ""
    elif case == "malformed_events":
        failure["_ng_failure_events"] = []

    def prepare():
        rows = [dict(identity)]
        return rows, RunManifest.create(source, rows, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    output = tmp_path / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        dispatched = store.pending(3)[0]
        store.record_dispatch(dispatched)
        store.record_outcome(dispatched | failure)
    reader = RolloutStore.read(output)
    assert len(reader.pending(3)) == expected_pending
    assert reader.attempt_count(identity) == 1
    assert reader.selected("failure")[0]["_ng_elapsed_s"] == 7.0
    with RolloutStore.start_or_resume(output, prepare, resume=True) as resumed:
        assert len(resumed.pending(3)) == expected_pending
    if not expected_pending:
        # An unfinished newer reservation cannot erase this recorded stop decision.
        with RolloutStore.start_or_resume(output, prepare, resume=True) as resumed:
            resumed.record_dispatch(dispatched | {"_ng_attempt_index": 1})
        assert RolloutStore.read(output).pending(3) == []
        assert RolloutStore.read(output).attempt_count(identity) == 1

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    config = RolloutCollectionConfig(
        agent_name="agent",
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(legacy / "rollouts.jsonl"),
        resume_from_cache=True,
    )
    config.materialized_jsonl_fpath.write_text(json.dumps(identity) + "\n")
    (legacy / "rollouts.jsonl").write_text("")
    sidecar = legacy / "rollouts_failures.jsonl"
    sidecar.write_text(json.dumps(identity | failure) + "\n")
    before = sidecar.read_bytes()
    pending, *_ = RolloutCollectionHelper()._load_from_cache(config)
    assert len(pending) == expected_pending
    assert sidecar.read_bytes() == before
