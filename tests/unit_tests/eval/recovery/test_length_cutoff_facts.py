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
"""A policy call that ended on a length cutoff is counted in the gate's facts, whatever the rollout did next."""

from nemo_gym.eval.recovery.gate import failure_facts, gate_summary_from_collected
from nemo_gym.eval.recovery.planner import decide_from_collected
from tests.unit_tests.eval.recovery.test_repetition_stop_facts import (
    ORDINARY,
    POLICY,
    answered,
    call,
    collect,
    judged_by,
    with_calls,
)


def cut(index: int, reason: str = "length", model: str = POLICY) -> dict:
    found = call(index, ORDINARY, model=model)
    found["response_metadata"]["finish_reason"] = reason
    return found


def test_the_facts_count_the_policy_calls_cut_off_at_length_and_their_rows(tmp_path) -> None:
    collected = collect(
        tmp_path,
        [
            # Cut turns the rollout carried on from: the final answer is complete, so no answer is truncated.
            with_calls(answered(0), cut(0), call(1, ORDINARY), cut(2, "max_output_tokens")),
            with_calls(answered(1), call(0, ORDINARY), cut(1, "max_tokens")),
            with_calls(answered(2), call(0, ORDINARY)),
            # A judge's cutoff is the judge's, not the evaluated model's.
            with_calls(answered(3), call(0, ORDINARY), cut(1, model="judge_model")),
        ],
    )
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    facts = failure_facts(decision, collected)

    assert facts["length_cutoffs"] == {"count": 3, "rows": 2}
    assert facts["response_failures"]["by_cause"]["truncated"]["count"] == 0
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert "length cutoffs: 3 policy calls in 2 trials" in task["health_verdict"]


def test_a_judge_on_the_policy_server_is_not_the_policy_cut_off(tmp_path) -> None:
    collected = collect(
        tmp_path,
        [
            judged_by(with_calls(answered(0), call(0, ORDINARY), cut(1)), {"model_call_id": "c1"}),
            judged_by(with_calls(answered(1), cut(0), call(1, ORDINARY)), {"model_call_id": "c1"}),
        ],
    )
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)

    assert failure_facts(decision, collected)["length_cutoffs"] == {"count": 1, "rows": 1}


def test_a_run_without_length_cutoffs_counts_none_and_says_nothing(tmp_path) -> None:
    collected = collect(tmp_path, [with_calls(answered(0), call(0, ORDINARY)), answered(1)])
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)

    assert failure_facts(decision, collected)["length_cutoffs"] == {"count": 0, "rows": 0}
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert "length cutoffs" not in task["health_verdict"]
