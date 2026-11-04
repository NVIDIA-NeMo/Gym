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
"""A benchmark's counted zero after the agent's own sandbox ended at its memory limit is the model's result in the
trial gate; any other sandbox OOM stays a provider fault."""

import json

import pytest

from nemo_gym.core import failure_kinds
from nemo_gym.orchestration.recovery import fault_class
from nemo_gym.trajectory.trajectory_record import NG_AGENT_SANDBOX_OBSERVATION_KEY
from tests.unit_tests.test_evaluation_health import _gate, _run


def _oom_zero(row: dict, *, role: str = "agent", failure_id: str = "oom-1", selected: str | None = None) -> None:
    """The row DeepSWE saves for an agent sandbox OOM its policy counts as zero: an empty answer scored zero, the
    selected kind ``provider_oom_killed``, and the agent's own sandbox observation."""
    observation = {
        "kind": "sandbox",
        "role": role,
        "provider": "opensandbox",
        "sandbox_id": "sbx-1",
        "phase": "agent_execution",
        "outcome": "oom",
        "failure_id": failure_id,
    }
    row["reward"] = 0.0
    row["failure_kind"] = failure_kinds.PROVIDER_OOM_KILLED
    row["mask_sample"] = False
    row["response"]["output"] = []
    row["response"]["metadata"] = dict(row["response"].get("metadata") or {})
    row["response"]["metadata"][NG_AGENT_SANDBOX_OBSERVATION_KEY] = json.dumps(observation)
    row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    if selected is not None:
        row["_ng_failure_id"] = selected


def _gated(**oom):
    collected = _run(10)
    _oom_zero(collected.results[-1], **oom)
    _, verdict, _ = _gate(collected)
    return verdict


def test_an_agent_sandbox_oom_zero_is_the_models_result_and_does_not_fail_the_gate():
    passed, reason, report = _gated()
    assert passed is True, reason
    assert report["evaluation"]["count"] == 0 and report["evaluation"]["server"] == 0
    assert report["model"]["kinds"] == {failure_kinds.AGENT_SANDBOX_OOM: 1}
    assert "model faults: 1 of 10 (10.00%) (agent_sandbox_oom 1)" in reason


@pytest.mark.parametrize(
    "oom",
    [{"role": "verifier"}, {"role": "environment"}, {"failure_id": "oom-1", "selected": "another-event"}],
    ids=["verifier sandbox", "environment sandbox", "observation of another event"],
)
def test_any_other_sandbox_oom_stays_a_provider_fault(oom):
    passed, reason, report = _gated(**oom)
    assert passed is False
    assert report["evaluation"]["kinds"] == {failure_kinds.PROVIDER_OOM_KILLED: 1}
    assert report["evaluation"]["server"] == 1 and report["model"]["count"] == 0
    assert "server 1: provider_oom_killed 1" in reason


def test_the_agent_sandbox_oom_class_is_the_models_and_no_producer_may_emit_it():
    assert fault_class(failure_kinds.AGENT_SANDBOX_OOM) == "model"
    assert failure_kinds.AGENT_SANDBOX_OOM in failure_kinds.GATE_DERIVED_KINDS
    assert failure_kinds.AGENT_SANDBOX_OOM not in failure_kinds.FAILURE_KINDS
    with pytest.raises(ValueError, match="derived by the acceptance gate"):
        failure_kinds.require_failure_kind(failure_kinds.AGENT_SANDBOX_OOM)
