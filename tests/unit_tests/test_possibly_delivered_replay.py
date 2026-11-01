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

from nemo_gym.rollout_recovery import RunManifest, has_unclassified_execution_failure
from nemo_gym.rollout_store import RolloutStore


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
