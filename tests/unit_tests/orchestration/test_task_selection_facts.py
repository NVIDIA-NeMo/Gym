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
"""The inspect of a run that selected its tasks by id labels its score as a subset score."""

import json
import subprocess
import sys

from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    decide_from_collected,
    gate_summary_from_collected,
)
from tests.unit_tests.orchestration.test_eval_recovery_gate import inventory, scored


DIGEST = "ab" * 32
RECORD = {"source": "subset_v1.txt", "tasks": 2, "rows": 2, "task_ids_sha256": DIGEST, "task_ids": ["0", "1"]}


def inspect(tmp_path, selection: str | None) -> dict:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inventory(2)))
    (artifacts / "rollouts.jsonl").write_text("".join(json.dumps(scored(t)) + "\n" for t in range(2)))
    if selection is not None:
        (artifacts / "rollouts_task_selection.json").write_text(selection)
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    collected = CollectedEvidence.model_validate_json(raw.stdout)
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    return gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)


def test_a_subset_run_is_labelled_with_its_source_counts_and_digest(tmp_path) -> None:
    task = inspect(tmp_path, json.dumps(RECORD))

    # The ids stay on the run host; the digest names the list.
    assert task["task_selection"] == {"source": "subset_v1.txt", "tasks": 2, "rows": 2, "task_ids_sha256": DIGEST}
    assert task["warnings"][0] == (
        "subset run: 2 tasks (2 rows) selected by task id from subset_v1.txt (task_ids sha256 abababababab); "
        "its score is a subset score, report it under the subset's name, never as the benchmark's"
    )


def test_an_unreadable_selection_record_still_labels_the_run_a_subset(tmp_path) -> None:
    task = inspect(tmp_path, "{not json")

    assert task["task_selection"] == {"source": None, "tasks": None, "rows": None, "task_ids_sha256": None}
    assert task["warnings"][0].startswith("subset run: the task selection record is unreadable")


def test_a_run_of_every_task_has_no_subset_label(tmp_path) -> None:
    task = inspect(tmp_path, None)

    assert task["task_selection"] is None
    assert not any(warning.startswith("subset run") for warning in task["warnings"])
