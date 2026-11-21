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
"""The recovery gate names a partial score (max_failed_rollout_fraction) read from the run's aggregate."""

import json
import subprocess
import sys

from nemo_gym.eval.recovery.evidence import COLLECT_SCRIPT, CollectedEvidence
from nemo_gym.eval.recovery.gate import gate_summary_from_collected
from nemo_gym.eval.recovery.planner import decide_from_collected
from tests.unit_tests.eval.recovery.test_eval_recovery_gate import infra_failure, inventory, scored


def test_the_gate_names_a_partial_score_and_never_passes_it(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inventory(4)))
    (artifacts / "rollouts.jsonl").write_text("".join(json.dumps(scored(t)) + "\n" for t in range(3)))
    (artifacts / "rollouts_failures.jsonl").write_text(json.dumps(infra_failure(3, attempt=0)) + "\n")
    record = {
        "accepted_failures": 1,
        "expected": 4,
        "accepted_failure_fraction": 0.25,
        "max_failed_rollout_fraction": 0.3,
    }

    def gate(mark):
        entry = {"agent_ref": {"name": "a"}, "agent_metrics": {"mean/reward": 0.75}, "key_metrics": {}}
        aggregate = [entry | ({"_ng_partial_score": mark} if mark is not None else {})]
        (artifacts / "rollouts_aggregate_metrics.json").write_text(json.dumps(aggregate))
        raw = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                COLLECT_SCRIPT,
                str(tmp_path),
                '["agent_metrics", "mean/reward"]',
                '{"agent": "a"}',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        collected = CollectedEvidence.model_validate_json(raw.stdout)
        d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
        return gate_summary_from_collected(d, collected, max_unhealthy_fraction=None)

    task = gate(record)
    assert task["partial_score"] == record
    assert task["warnings"][0] == (
        "partial score: 1 of 4 rollouts (25.00%) were unsettled and counted as zero, within "
        "max_failed_rollout_fraction 0.3; it is not a full score"
    )
    assert task["status"] != "SUCCESS"
    # A full score carries no record, and a malformed one is not read as a partial score.
    for mark in (None, record | {"accepted_failures": -1}):
        task = gate(mark)
        assert task["partial_score"] is None
        assert not any(w.startswith("partial score") for w in task["warnings"])
