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
"""A possibly delivered request is replayed only when its row keeps no response to duplicate."""

import json

import pytest

from nemo_gym.rollouts.rollout_recovery import RunManifest, has_unclassified_execution_failure
from nemo_gym.rollouts.rollout_store import RolloutStore


def _lost_connection_row(**extra):
    """The collector's sidecar row for a /run whose connection dropped before any response arrived."""
    return {
        "_ng_failure_class": "agent_request_failed",
        "_ng_failure_type": "ServerDisconnectedError",
        "_ng_failure_message": "Server disconnected",
        "_ng_failure_record": {
            "schema_version": 1,
            "episode_id": {"rollout_id": "0-0", "attempt": 0},
            "run_id": "run",
            "source": "collector",
            "delivery": "possibly_delivered",
            "failure": {
                "failure_reason": "Server disconnected",
                "terminal": False,
                "failure_kind": "agent_request_failed",
            },
            "http_status": None,
            "exception_type": "ServerDisconnectedError",
        },
        "_ng_failure_events": {
            "dispatch": {
                "event_id": "dispatch",
                "boundary": "collection",
                "failure_kind": "collection_dispatch_failed",
                "operation": "collection.run",
            }
        },
        "_ng_failure_id": "dispatch",
        **extra,
    }


def test_a_lost_connection_with_no_saved_response_may_run_again():
    assert not has_unclassified_execution_failure(_lost_connection_row())
    assert not has_unclassified_execution_failure(_lost_connection_row(_ng_unrouted_run_error=True))


@pytest.mark.parametrize("key", ["response", "_ng_failure_response", "_ng_failure_partial_response"])
def test_a_possibly_delivered_request_that_kept_a_response_never_runs_again(key):
    response = {"object": "response", "id": "resp-1", "status": "incomplete", "output": []}
    assert has_unclassified_execution_failure(_lost_connection_row(**{key: response}))
    assert has_unclassified_execution_failure(_lost_connection_row(_ng_unrouted_run_error=True, **{key: response}))


def test_the_store_resumes_a_lost_connection_row_within_its_attempt_cap(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(rows[0]) + "\n")

    def prepare():
        return rows, RunManifest.create(source, rows, {}, {}).model_copy(update={"max_rollout_attempts": 2})

    output = artifacts / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for _ in range(2):
            row = dict(rows[0])
            store.allocate_attempt(row)
            failure = _lost_connection_row()
            failure["_ng_failure_record"] = failure["_ng_failure_record"] | {
                "run_id": store.manifest.run_id,
                "episode_id": {"rollout_id": "0-0", "attempt": row["_ng_attempt_index"]},
            }
            store.record_outcome(row | failure)
            if row["_ng_attempt_index"] == 0:
                assert [pending["_ng_attempt_index"] for pending in store.pending()] == [1]
    # The second lost connection spends the cap of two: nothing is retried without budget.
    assert RolloutStore.read(output).pending() == []


def _route_failure_row(**extra):
    """A legacy /run that answered 500 after the agent's sandbox ended: collection event caused by the agent's."""
    return {
        "_ng_failure_class": "agent_run_error",
        "_ng_failure_http_status": 500,
        "_ng_failure_record": {"source": "collector", "delivery": "delivered", "failure": {"terminal": False}},
        "_ng_failure_events": {
            "agent": {"boundary": "agent", "failure_kind": "agent_route_failed", "operation": "agent.run"},
            "dispatch": {
                "boundary": "collection",
                "failure_kind": "collection_dispatch_failed",
                "operation": "collection.run",
                "cause_id": "agent",
            },
        },
        "_ng_failure_id": "dispatch",
        **extra,
    }


_ANSWER = {
    "object": "response",
    "id": "resp-answer",
    "status": "completed",
    "output": [{"type": "message", "role": "assistant", "status": "completed", "content": []}],
}


def test_a_route_failure_that_kept_no_answer_may_run_again():
    assert not has_unclassified_execution_failure(_route_failure_row())


def test_a_route_failure_that_kept_an_answer_never_runs_again():
    assert has_unclassified_execution_failure(_route_failure_row(_ng_failure_response=_ANSWER))


@pytest.mark.parametrize("status,refused", [(500, False), (503, False), (400, True), (None, True)])
def test_a_legacy_run_error_without_an_agent_event_reruns_only_after_a_5xx(status, refused):
    row = _route_failure_row(_ng_failure_http_status=status)
    del row["_ng_failure_events"]["agent"]
    del row["_ng_failure_events"]["dispatch"]["cause_id"]
    assert has_unclassified_execution_failure(row) is refused


def test_an_omitted_failure_report_never_blocks_resume_by_itself():
    # The transport's size bound dropped the report: its events say nothing either way, so the row's own record decides.
    omitted = _route_failure_row(_ng_failure_events_omitted=True)
    assert not has_unclassified_execution_failure(omitted)
    # The selected event may be one the bound dropped: still no veto from the transport.
    partial = _route_failure_row(_ng_failure_events_omitted=True, _ng_failure_id="dropped")
    assert not has_unclassified_execution_failure(partial)
    # A kept answer is still never run again.
    assert has_unclassified_execution_failure(
        _lost_connection_row(_ng_failure_events_omitted=True, _ng_failure_response=_ANSWER)
    )


def test_a_judge_failure_resume_leaves_to_recovery_is_counted_and_named(tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0}, {"_ng_task_index": 1, "_ng_rollout_index": 0}]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        return rows, RunManifest.create(source, rows, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    output = artifacts / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for index, failure_class in ((0, "judge_failed"), (1, "agent_request_failed")):
            row = dict(rows[index])
            store.allocate_attempt(row)
            failure = _lost_connection_row(_ng_failure_class=failure_class)
            failure["_ng_failure_record"] = failure["_ng_failure_record"] | {
                "run_id": store.manifest.run_id,
                "episode_id": {"rollout_id": f"{index}-0", "attempt": row["_ng_attempt_index"]},
                "failure": failure["_ng_failure_record"]["failure"] | {"failure_kind": failure_class},
            }
            store.record_outcome(row | failure)
    reader = RolloutStore.read(output)
    capsys.readouterr()
    # The lost connection is run again; the judge failure is left to verification recovery, and says so.
    assert [row["_ng_task_index"] for row in reader.pending()] == [1]
    assert "1 failed rollout(s) are not run again" in capsys.readouterr().out
    coverage = reader.coverage()
    assert coverage["stopped_not_retried"] == 1 and coverage["retryable"] == 1
    # It is one of the final failures resume does not run again, now named apart from terminal ones.
    assert coverage["failed_final"] == 1 and coverage["failed_not_final"] == 1
    # Only the retryable one is open: the stopped one is final for resume, so open cannot stay above zero forever.
    assert coverage["open"] == 1


@pytest.mark.parametrize("failure_class", ["judge_failed", "judge_invalid", "judge_unparseable"])
def test_every_judge_failure_is_left_to_verification_recovery(failure_class):
    # The registered judge_unparseable is a judge failure too: its answer is judged again, never regenerated.
    assert has_unclassified_execution_failure(_route_failure_row(_ng_failure_class=failure_class))


def test_an_omitted_report_does_not_lift_a_terminal_failure(tmp_path):
    # The omitted flag only stops vetoing a retry; a failure the producer marked terminal still stays final.
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(rows[0]) + chr(10))

    def prepare():
        return rows, RunManifest.create(source, rows, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    output = artifacts / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = dict(rows[0])
        store.allocate_attempt(row)
        failure = _lost_connection_row(_ng_failure_events_omitted=True, _ng_failure_terminal=True)
        failure["_ng_failure_record"] = failure["_ng_failure_record"] | {
            "run_id": store.manifest.run_id,
            "episode_id": {"rollout_id": "0-0", "attempt": row["_ng_attempt_index"]},
            "failure": failure["_ng_failure_record"]["failure"] | {"terminal": True},
        }
        store.record_outcome(row | failure)
    reader = RolloutStore.read(output)
    assert not has_unclassified_execution_failure(failure)
    assert reader.pending() == [] and reader.coverage()["stopped_not_retried"] == 0
