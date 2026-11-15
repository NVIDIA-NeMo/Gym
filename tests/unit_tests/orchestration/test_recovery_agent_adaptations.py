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
"""A row's ``agent_adaptations`` counts are summed into the run's failure facts."""

import json
import subprocess
import sys

from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    decide_from_collected,
    failure_facts,
)
from tests.unit_tests.orchestration.test_eval_recovery_gate import inventory, scored


def adapted(task: int, rejections: int, regenerations: int) -> dict:
    counts = {"input_limit_rejections": rejections, "output_truncation_regenerations": regenerations}
    return scored(task) | {"agent_adaptations": counts}


def facts_of(collected: CollectedEvidence) -> dict:
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    return failure_facts(d, collected)


def test_counts_are_summed_with_the_rows_that_adapted() -> None:
    collected = CollectedEvidence(
        materialized=inventory(4),
        results=[adapted(0, 2, 0), adapted(1, 1, 1), adapted(2, 0, 0), scored(3)],
        failures=[],
    )
    assert facts_of(collected)["agent_adaptations"] == {
        "input_limit_rejections": {"count": 3, "rows": 2},
        "output_truncation_regenerations": {"count": 1, "rows": 1},
    }


def test_no_adapting_agent_reports_none() -> None:
    collected = CollectedEvidence(materialized=inventory(2), results=[scored(0), scored(1)], failures=[])
    assert facts_of(collected)["agent_adaptations"] == {}


def test_malformed_counts_are_ignored() -> None:
    rows = [
        scored(0) | {"agent_adaptations": {"a": -1, "b": True, "c": "2", "d": 1.5}},
        scored(1) | {"agent_adaptations": [1]},
    ]
    collected = CollectedEvidence(materialized=inventory(2), results=rows, failures=[])
    assert facts_of(collected)["agent_adaptations"] == {}


def test_the_collector_carries_the_counts_off_the_run_host(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [adapted(0, 2, 0), adapted(1, 0, 1)]
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inventory(2)))
    (artifacts / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    collected = CollectedEvidence.model_validate_json(raw.stdout)
    assert facts_of(collected)["agent_adaptations"] == {
        "input_limit_rejections": {"count": 2, "rows": 1},
        "output_truncation_regenerations": {"count": 1, "rows": 1},
    }
