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
"""A finished run whose verifier recorded no grading operation is observed when its saved output re-derives the grade.

Two verifiers save enough to re-derive it, read without a GPU from the saved run:

- a node-set F1 verifier saves the predicted nodes, the parse outcome and ``f1`` (its reward); the collector re-parses
  the saved answer's final line and re-scores it against the saved expected nodes (``node_set_f1``);
- a sandboxed task verifier writes its own ``reward.json`` per session beside the artifacts; the row's reward must be
  that file's ``reward`` (``session_reward_file``). Test counts and partial credit are not the grade: a task can pass
  every counted test and still score 0.

The fixtures are graded rows of saved runs, cut to the fields the predicates read, with ids and answer text replaced.
"""

import json
from pathlib import Path

import pytest

from nemo_gym.orchestration.recovery import (
    CollectedEvidence,
    decide_from_collected,
    grading_artifact,
    trial_verdict,
)
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir


FIXTURES = Path(__file__).parent / "fixtures"
NODE_ROWS = json.loads((FIXTURES / "saved-verifier-output-node-f1.json").read_text())
SESSION_ROWS = json.loads((FIXTURES / "saved-verifier-output-session-reward.json").read_text())
GATE_FIXTURE = FIXTURES / "cpu-grading-mcqa-healthy.json"


def _row(saved: dict, task: int = 0, **changes) -> dict:
    row = {k: v for k, v in saved.items() if not k.startswith("_")} | changes
    return ident(task, **row)


def _collect(tmp_path, rows: list[dict], reward_files: dict[str, str] | None = None) -> list[dict]:
    job = write_job_dir(tmp_path / "job", n=len(rows), results=rows, failures=[])
    for session, text in (reward_files or {}).items():
        directory = job / "resources" / session / "verifier"
        directory.mkdir(parents=True)
        (directory / ("reward.txt" if not text.startswith("{") else "reward.json")).write_text(text)
    return collect_with_score(job, "[]").results


def _reward_file(saved: dict) -> dict[str, str]:
    return {saved["session_id"]: json.dumps(saved["_reward_file"])}


@pytest.mark.parametrize("saved", NODE_ROWS, ids=[row["_class"] for row in NODE_ROWS])
def test_a_saved_node_f1_the_answer_reproduces_records_the_grading(tmp_path, saved):
    [row] = _collect(tmp_path, [_row(saved)])
    assert row["_ng_collected_grading_facts"]["node_f1"] == {
        "f1": saved["f1"],
        "regraded": saved["f1"],
        "parse_matches": True,
    }
    assert grading_artifact(row) == "node_set_f1"


def test_the_collector_projects_node_f1_scores_only(tmp_path):
    saved = next(row for row in NODE_ROWS if row["_class"] == "partial")
    [row] = _collect(tmp_path, [_row(saved)])
    text = json.dumps(row)
    assert "Final Answer" not in text and not any(node in text for node in saved["predicted_nodes"])


@pytest.mark.parametrize(
    "changes",
    [
        {"reward": 1.0},
        {"f1": 1.0},
        {"predicted_nodes": ["0000000000"]},
        {"parse_failed": True},
        {"expected_answer": "[]"},
    ],
    ids=["reward-not-f1", "saved-f1-not-regraded", "nodes-not-reparsed", "parse-outcome-differs", "other-expected"],
)
def test_a_saved_node_f1_the_answer_does_not_reproduce_stays_unobserved(tmp_path, changes):
    saved = next(row for row in NODE_ROWS if row["_class"] == "partial")
    [row] = _collect(tmp_path, [_row(saved, **changes)])
    assert grading_artifact(row) is None


def test_an_unbound_answer_is_not_recorded_by_its_node_f1(tmp_path):
    saved = next(row for row in NODE_ROWS if row["_class"] == "full")
    [row] = _collect(tmp_path, [_row(saved, response=saved["response"] | {"id": None})])
    assert grading_artifact(row) is None


@pytest.mark.parametrize("saved", SESSION_ROWS, ids=[row["_class"] for row in SESSION_ROWS])
def test_the_session_reward_file_records_the_grading(tmp_path, saved):
    [row] = _collect(tmp_path, [_row(saved)], _reward_file(saved))
    assert row["_ng_collected_grading_facts"]["session_reward"] == saved["_reward_file"]["reward"]
    assert grading_artifact(row) == "session_reward_file"


def test_passing_every_counted_test_is_not_the_grade(tmp_path):
    # The verifier's own reward file says 0 although its one counted test passed: the row is observed at 0.
    saved = next(row for row in SESSION_ROWS if row["_class"] == "zero-with-passes")
    assert saved["num_tests_passed"] == saved["num_tests_total"] and saved["reward"] == 0.0
    [row] = _collect(tmp_path, [_row(saved)], _reward_file(saved))
    assert grading_artifact(row) == "session_reward_file"
    [row] = _collect(tmp_path / "other", [_row(saved, reward=1.0)], _reward_file(saved))
    assert grading_artifact(row) is None


def test_a_plain_reward_txt_is_read_when_there_is_no_reward_json(tmp_path):
    saved = next(row for row in SESSION_ROWS if row["_class"] == "full")
    [row] = _collect(tmp_path, [_row(saved)], {saved["session_id"]: "1.0"})
    assert grading_artifact(row) == "session_reward_file"


@pytest.mark.parametrize(
    "changes,files",
    [
        ({}, {}),
        ({}, {"tb4-other": json.dumps({"reward": 1.0})}),
        ({}, None),
        ({"evaluation_completed": False}, None),
        ({"infrastructure_error": "Agent infrastructure failure"}, None),
        ({"session_id": "../tb4-00000000000000000000000000000003"}, None),
    ],
    ids=["no-file", "another-session", "file-disagrees", "not-completed", "infrastructure-error", "path-escape"],
)
def test_a_row_its_session_reward_file_does_not_record_stays_unobserved(tmp_path, changes, files):
    saved = next(row for row in SESSION_ROWS if row["_class"] == "full")
    if files is None:
        files = {saved["session_id"]: json.dumps({"reward": 0.0 if not changes else 1.0})}
    [row] = _collect(tmp_path, [_row(saved, **changes)], files)
    assert grading_artifact(row) is None


@pytest.mark.parametrize("artifact", ["node_f1", "session_reward", None])
def test_the_trial_gate_reports_grading_derived_from_saved_verifier_output(artifact):
    collected = CollectedEvidence.model_validate_json(GATE_FIXTURE.read_text())
    row = collected.results[0]
    row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    reward = row["reward"]
    if artifact == "node_f1":
        row["_ng_collected_grading_facts"] = {"node_f1": {"f1": reward, "regraded": reward, "parse_matches": True}}
    elif artifact == "session_reward":
        row["evaluation_completed"] = True
        row["_ng_collected_grading_facts"] = {"session_reward": reward}
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    passed, reason, evaluation = trial_verdict(decision, collected)
    if artifact is None:
        assert passed is None and evaluation["exact"] is False, reason
        return
    name = {"node_f1": "node_set_f1", "session_reward": "session_reward_file"}[artifact]
    assert passed is True and evaluation["exact"] is True, reason
    assert evaluation["facts"]["grading_derived_from_saved_output"] == {name: 1}
    assert f"grading derived from saved verifier output: {name} 1" in reason
