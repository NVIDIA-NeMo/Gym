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
"""Evidence builders the recovery and orchestration tests share."""

import json
from pathlib import Path

from nemo_gym.eval.recovery.evidence import CollectedEvidence
from nemo_gym.eval.recovery.gate import gate_summary_from_collected, trial_verdict
from nemo_gym.eval.recovery.planner import decide_from_collected
from tests.helpers.recovery_evidence import final_model_zero


FIXTURES = Path(__file__).resolve().parents[1] / "unit_tests/eval/recovery/fixtures"
FIXTURE = FIXTURES / "cpu-grading-mcqa-healthy.json"


IDS = (
    "resp_scripted_1",
    "cc9b71700e2e47bcb280552c053e1ba9",  # pragma: allowlist secret
    "db88801997f34c978b960e86a4abc542",  # pragma: allowlist secret
    "verify-a188f19c8d794e77bbae490042027a9c",
    "grade-bbef44c2491649afb24ebd10d87ad6d8",
)


def _run(scored: int, *, settled: dict[str, int] | None = None) -> CollectedEvidence:
    """``scored`` graded trials (reward 1) and, after them, settled counts-as-zero failures by kind."""
    base = json.loads(FIXTURE.read_text())
    template = json.dumps(base["results"][0])
    rows, materialized, selection, failures = [], [], [], []
    for task in range(scored):
        text = template.replace('"task_id": "0"', f'"task_id": "{task}"').replace('"0-0"', f'"{task}-0"')
        for value in IDS:
            text = text.replace(value, f"{value}-{task}")
        row = json.loads(text)
        row["_ng_task_index"], row["_line"] = task, task
        rows.append(row)
    task = scored
    for kind, count in (settled or {}).items():
        for _ in range(count):
            failure = final_model_zero(task, 0)
            failure["failure_kind"] = kind
            failure["_ng_failure_events"][f"f-{task}-0"]["failure_kind"] = kind
            failure["_line"] = len(failures)
            failures.append(failure)
            task += 1
    for index in range(task):
        materialized.append({"_ng_task_index": index, "_ng_rollout_index": 0, "_ng_collected_task_id": str(index)})
        selection.append(
            {"identity": [index, 0], "attempts": 0, "max_attempts": 3, "pending": False, "reserved": True}
        )
    base |= {"results": rows, "materialized": materialized, "managed_selection": selection, "failures": failures}
    base["quality"]["artifacts"]["expected"] = task
    return CollectedEvidence.model_validate(base)


def _gate(collected: CollectedEvidence):
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    return decision, trial_verdict(decision, collected), task


def recorded_run(name: str) -> str:
    """A recorded run's collected evidence as JSON: its file, or its base file with the values its patch sets."""
    path = FIXTURES / f"{name}.json"
    if path.exists():
        return path.read_text()
    patch = json.loads((FIXTURES / f"{name}.patch.json").read_text())
    run = json.loads((FIXTURES / patch["base"]).read_text())
    for keys, value in patch["set"]:
        node = run
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return json.dumps(run)


def fixture(case="healthy"):
    return CollectedEvidence.model_validate_json(recorded_run(f"cpu-tool-trial-{case}"))
