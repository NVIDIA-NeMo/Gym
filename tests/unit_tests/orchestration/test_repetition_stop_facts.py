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
"""A policy call the provider stopped for repetition is counted in the gate's facts, and scored as answered."""

import json
import subprocess
import sys

from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    decide_from_collected,
    failure_facts,
    gate_summary_from_collected,
)
from tests.unit_tests.orchestration.test_eval_recovery_gate import ident, inventory


POLICY = "policy_model"
# What Gym saves for a call vLLM stopped with its repetition detection: the SDK's `stop` with the provider's own
# reason beside it, on a Responses reply and on a Chat Completions reply.
RESPONSES_REPETITION = {"status": "completed", "native_finish_reason": "repetition"}
CHAT_REPETITION = {"choices": [{"finish_reason": "stop", "native_finish_reason": "repetition"}]}
CHAT_STOP_REASON_ONLY = {"choices": [{"finish_reason": "stop", "stop_reason": "repetition_detected"}]}
ORDINARY = {"status": "completed"}


def answered(task: int, reward: float = 1.0) -> dict:
    message = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": f"answer {task}"}]}
    return ident(task, reward=reward, response={"status": "completed", "output": [message]})


def call(index: int, response: dict, model: str = POLICY) -> dict:
    return {
        "model_call_id": f"c{index}",
        "response": response,
        "response_metadata": {
            "status_code": 200,
            "finish_reason": "stop",
            "model_ref": {"type": "responses_api_models", "name": model},
        },
    }


def with_calls(row: dict, *calls: dict) -> dict:
    trajectory = {"task_id": str(row["_ng_task_index"]), "rollout_id": "r", "model_calls": list(calls)}
    return row | {"ng_policy_model_server": POLICY, "ng_trajectory": trajectory}


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


def test_the_facts_count_the_policy_calls_stopped_for_repetition_and_their_rows(tmp_path) -> None:
    collected = collect(
        tmp_path,
        [
            with_calls(answered(0, reward=0.0), call(0, RESPONSES_REPETITION), call(1, CHAT_REPETITION)),
            with_calls(answered(1), call(0, ORDINARY), call(1, CHAT_STOP_REASON_ONLY)),
            with_calls(answered(2), call(0, ORDINARY)),
            # A judge's repetition stop is the judge's, not the evaluated model's ending.
            with_calls(answered(3), call(0, ORDINARY), call(1, CHAT_REPETITION, model="judge_model")),
        ],
    )
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    facts = failure_facts(decision, collected)

    assert facts["repetition_stops"] == {"count": 3, "rows": 2}
    # The model's own ending: the answer it produced is scored as it is, so it is no response or trial failure.
    assert facts["response_failures"]["count"] == 0
    assert facts["trial_failures"]["count"] == 0
    # Only the flag leaves the run host, not the saved response.
    [first] = collected.results[0]["ng_trial_observations"]["model_calls"][:1]
    assert first["repetition_stop"] is True
    assert "response" not in first

    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert "repetition stops: 3 policy calls in 2 trials" in task["health_verdict"]


def test_a_run_without_repetition_stops_counts_none_and_says_nothing(tmp_path) -> None:
    collected = collect(tmp_path, [with_calls(answered(0), call(0, ORDINARY)), answered(1)])
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)

    assert failure_facts(decision, collected)["repetition_stops"] == {"count": 0, "rows": 0}
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert "repetition" not in task["health_verdict"]


def judged_by(row: dict, *refs: dict) -> dict:
    """The row with one verifier invocation that owns the calls ``refs`` name."""
    invocation = {"invocation_id": "v0", "status": "completed", "model_calls": list(refs)}
    return row | {"ng_trajectory": row["ng_trajectory"] | {"verifier_invocations": [invocation]}}


def test_a_judge_on_the_policy_server_is_not_the_policy_looping(tmp_path) -> None:
    # Many judges are served by the policy's own server: the verifier's ownership, not the server name, says whose
    # call it is.
    judge_by_id = call(1, CHAT_REPETITION)
    judge_by_response = call(1, CHAT_REPETITION) | {"model_call_id": None}
    judge_by_response["response_metadata"] = judge_by_response["response_metadata"] | {"response_id": "resp-j"}
    collected = collect(
        tmp_path,
        [
            judged_by(with_calls(answered(0), call(0, ORDINARY), judge_by_id), {"model_call_id": "c1"}),
            judged_by(
                with_calls(answered(1), call(0, ORDINARY), judge_by_response),
                {"response_id": "resp-j", "model_ref": {"type": "responses_api_models", "name": POLICY}},
            ),
            # The policy's own loop in a judged row still counts.
            judged_by(
                with_calls(answered(2), call(0, RESPONSES_REPETITION), call(1, ORDINARY)), {"model_call_id": "c1"}
            ),
        ],
    )
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)

    assert failure_facts(decision, collected)["repetition_stops"] == {"count": 1, "rows": 1}


def test_only_the_row_that_decides_the_trial_is_counted(tmp_path) -> None:
    # A rollout saved twice (an attempt a later one superseded) is one trial: only its last row counts.
    first = with_calls(answered(0, reward=0.0), call(0, RESPONSES_REPETITION), call(1, CHAT_REPETITION))
    last = with_calls(answered(0), call(0, RESPONSES_REPETITION))
    refit = {"requested_max_tokens": 131072, "window": 8192, "applied_max_tokens": 8076}
    for row in (first, last):
        row["ng_trajectory"]["model_calls"][0]["response_metadata"]["output_budget_refit"] = refit
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text(json.dumps(inventory(1)[0]) + "\n")
    (artifacts / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (first, last)))
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    collected = CollectedEvidence.model_validate_json(raw.stdout)
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    facts = failure_facts(decision, collected)

    assert facts["repetition_stops"] == {"count": 1, "rows": 1}
    assert facts["output_budget_refits"] == {"count": 1, "rows": 1}
