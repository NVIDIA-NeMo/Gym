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
"""A model call sent again with its output budget lowered to fit the context window is counted in the gate's facts."""

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


REFIT = {"requested_max_tokens": 131072, "window": 8192, "applied_max_tokens": 8076}


def with_calls(row: dict, *refits: dict | None) -> dict:
    calls = [
        {"model_call_id": f"c{index}", "response_metadata": {"status_code": 200, "finish_reason": "stop"}}
        for index in range(len(refits))
    ]
    for call, refit in zip(calls, refits):
        if refit is not None:
            call["response_metadata"]["output_budget_refit"] = refit
    return row | {"ng_trajectory": {"task_id": str(row["_ng_task_index"]), "rollout_id": "r", "model_calls": calls}}


def collect(tmp_path, results: list[dict]) -> CollectedEvidence:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in inventory(len(results)))
    )
    (artifacts / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    return CollectedEvidence.model_validate_json(raw.stdout)


def test_the_facts_count_refitted_calls_and_the_rows_that_made_them(tmp_path) -> None:
    collected = collect(
        tmp_path,
        [
            with_calls(scored(0), REFIT, None, REFIT),
            with_calls(scored(1), None),
            with_calls(scored(2), REFIT),
        ],
    )
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)

    assert failure_facts(decision, collected)["output_budget_refits"] == {"count": 3, "rows": 2}
    # The collector carries the event's three numbers off the run host.
    [first] = [
        call for call in collected.results[0]["ng_trial_observations"]["model_calls"] if call["model_call_id"] == "c0"
    ]
    assert first["response_metadata"]["output_budget_refit"] == REFIT


def test_a_run_without_refits_counts_none(tmp_path) -> None:
    collected = collect(tmp_path, [with_calls(scored(0), None), scored(1)])
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)

    assert failure_facts(decision, collected)["output_budget_refits"] == {"count": 0, "rows": 0}
