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
"""An artifact the recovery collector cannot read is recorded as unknown; it never crashes the collector."""

import json
import os
import subprocess
import sys

import pytest

from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    RecoveryAction,
    decide_from_collected,
    gate_summary_from_collected,
)


pytestmark = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root reads a file whatever its mode"
)


def _run(job, *, unreadable: str):
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True)
    inventory = [{"_ng_task_index": t, "_ng_rollout_index": 0} for t in range(2)]
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inventory))
    (artifacts / "rollouts.jsonl").write_text(
        "".join(json.dumps(r | {"reward": 1.0, "response": {"output": []}}) + "\n" for r in inventory)
    )
    (artifacts / "rollouts_failures.jsonl").write_text("")
    (artifacts / "rollouts_aggregate_metrics.json").write_text(
        json.dumps([{"agent_ref": {"name": "a"}, "agent_metrics": {"mean/reward": 1.0}, "key_metrics": {}}])
    )
    (artifacts / "quality_summary.json").write_text("{}")
    target = artifacts / unreadable
    target.chmod(0)
    try:
        raw = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                COLLECT_SCRIPT,
                str(job),
                json.dumps(["agent_metrics", "mean/reward"]),
                json.dumps({"agent": "a"}),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        target.chmod(0o600)
    return CollectedEvidence.model_validate_json(raw.stdout)


@pytest.mark.parametrize("name", ["rollouts_aggregate_metrics.json", "quality_summary.json"])
def test_an_unreadable_metric_or_summary_is_unknown_never_a_crash_or_a_pass(tmp_path, name):
    collected = _run(tmp_path, unreadable=name)
    assert collected.unreadable_files == [f"artifacts/{name}"]
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert task["status"] == "FAILED"
    assert f"unreadable artifacts, read as unknown: ['artifacts/{name}']" in task["errors"]
    if name == "rollouts_aggregate_metrics.json":
        assert collected.score is None
        assert (
            "score unavailable: no requested metric was read (pass --score-key, and --score-agent or --score-environment for a Gym aggregate)"
            in task["errors"]
        )


@pytest.mark.parametrize("name", ["rollouts.jsonl", "rollouts_failures.jsonl", "rollouts_materialized_inputs.jsonl"])
def test_unreadable_saved_rows_stop_the_decision_instead_of_reading_as_absent(tmp_path, name):
    collected = _run(tmp_path, unreadable=name)
    assert f"artifacts/{name}" in collected.unreadable_files
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert decision.action == RecoveryAction.STOP
    assert decision.reason == f"saved rows are unreadable ({name}), so what they hold is unknown"
