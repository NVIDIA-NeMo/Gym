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
"""The fault classes are one shared default in the failure vocabulary; a policy timeout's class is a run setting."""

import pytest

from nemo_gym.core import failure_kinds
from nemo_gym.orchestration import recovery
from nemo_gym.orchestration.recovery import (
    GateCriteria,
    fault_class,
    fault_report,
    gate_summary_from_collected,
    trial_verdict,
)
from tests.unit_tests.orchestration.test_eval_resubmit_decision import (  # noqa: F401
    complete_job,
    fake_executor,
    run_cli,
)
from tests.unit_tests.orchestration.test_evaluation_health import _gate, _run


def test_the_gate_uses_the_shared_class_lists():
    assert recovery.MODEL_FAULT_KINDS is failure_kinds.MODEL_FAULT_KINDS
    assert recovery.SERVER_FAULT_KINDS is failure_kinds.SERVER_FAULT_KINDS
    # The shared default covers what a policy-failure list counts as the policy's own.
    assert {
        "agent_timeout",
        "agent_command_not_allowed",
        "agent_limit",
        "agent_invalid_output",
        "model_context_limit",
    } <= failure_kinds.MODEL_FAULT_KINDS


def test_a_command_the_harness_refused_is_the_models_fault():
    from collections import Counter

    assert fault_class("agent_command_not_allowed") == "model"
    report = fault_report(Counter({"agent_command_not_allowed": 1}), 100)
    assert report["model"]["kinds"] == {"agent_command_not_allowed": 1}
    assert report["evaluation"]["count"] == 0


@pytest.mark.parametrize(
    "kind",
    [
        "truncated_answer",
        "empty_answer",
        "malformed_answer",
        "model_server_error",
        "non_policy_model_rejection",
        "invalid_output_budget",
        "invalid_request_parameter",
    ],
)
def test_the_gate_derived_classes_are_registered_but_no_producer_may_emit_them(kind):
    assert kind in failure_kinds.GATE_DERIVED_KINDS and failure_kinds.is_gate_derived(kind)
    assert kind not in failure_kinds.FAILURE_KINDS
    assert not failure_kinds.is_registered(kind)
    with pytest.raises(ValueError, match="derived by the acceptance gate"):
        failure_kinds.require_failure_kind(kind)


def _timed_out_run():
    """One trial's policy request timed out on every attempt; 99 others are graded."""
    return _run(99, settled={"model_policy_response_timeout": 1})


def test_a_policy_timeout_is_the_models_fault_by_default():
    collected = _timed_out_run()
    _, (passed, reason, report), task = _gate(collected)
    assert report["model"]["kinds"] == {"model_policy_response_timeout": 1}, reason
    assert report["evaluation"]["server"] == 0
    assert task["criteria"]["evaluation"]["policy_timeout_counts_as_model"] is True


def test_a_run_that_owns_its_serving_can_count_a_policy_timeout_as_a_server_fault():
    from nemo_gym.orchestration.recovery import decide_from_collected

    collected = _timed_out_run()
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    passed, reason, report = trial_verdict(decision, collected, policy_timeout_counts_as_model=False)
    assert report["evaluation"]["kinds"] == {"model_policy_response_timeout": 1}, reason
    assert report["evaluation"]["server"] == 1 and report["model"]["count"] == 0
    assert "(server 1: model_policy_response_timeout 1)" in reason
    task = gate_summary_from_collected(
        decision,
        collected,
        max_unhealthy_fraction=None,
        criteria=GateCriteria(policy_timeout_counts_as_model=False),
    )
    assert task["evaluation_health"]["evaluation"]["server"] == 1
    assert task["criteria"]["evaluation"]["policy_timeout_counts_as_model"] is False


def test_the_setting_is_a_strict_boolean():
    with pytest.raises(ValueError):
        GateCriteria(policy_timeout_counts_as_model="no")


@pytest.mark.parametrize("value,expected", [("false", False), ("true", True)])
def test_the_recover_command_sets_it_on_the_gate(tmp_path, fake_executor, capsys, value, expected):
    fake_executor.job_dir = complete_job(tmp_path)
    _, out = run_cli(
        ["eval", "resubmit", "gym-job-k", "bench", "--recover", "--policy-timeout-counts-as-model", value], capsys
    )
    assert out["gate_task"]["criteria"]["evaluation"]["policy_timeout_counts_as_model"] is expected


def test_ungraded_harness_failure_is_evaluation_side_not_model_invalid_output():
    from collections import Counter

    report = fault_report(Counter({"agent_route_failed": 1, "agent_invalid_output": 1}), 2)
    assert report["evaluation"]["kinds"] == {"agent_route_failed": 1}
    assert report["model"]["kinds"] == {"agent_invalid_output": 1}
