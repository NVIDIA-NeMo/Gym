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

"""``gym.rollout.completed_total``: the driver-side outcome counter, against a real in-memory reader."""

import pytest

from nemo_gym import rollout_collection
from nemo_gym.telemetry import gym_metrics
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics  # noqa: F401 - fixture


pytest.importorskip("opentelemetry.sdk.metrics")

COMPLETED = gym_metrics.ROLLOUT_COMPLETED_INSTRUMENT
OUTCOME = gym_metrics.ROLLOUT_OUTCOME_ATTRIBUTE
CLASS = gym_metrics.FAILURE_CLASS_ATTRIBUTE
REASON = gym_metrics.FAILURE_REASON_ATTRIBUTE


def _by_attrs(collected):
    return {tuple(sorted(p.attributes.items())): p.value for p in collected().get(COMPLETED, [])}


def test_scored_and_dropped_are_counted_with_class_and_reason(collected_metrics):  # noqa: F811
    gym_metrics.record_rollout_completed("scored")
    gym_metrics.record_rollout_completed("scored")
    gym_metrics.record_rollout_completed(
        "dropped", failure_class="infrastructure_error", failure_reason="SandboxTimeoutException"
    )
    gym_metrics.record_rollout_completed(
        "dropped", failure_class="infrastructure_error", failure_reason="RuntimeError"
    )
    gym_metrics.record_rollout_completed("dropped", failure_class="judge_failed")

    assert _by_attrs(collected_metrics) == {
        ((OUTCOME, "scored"),): 2,
        ((CLASS, "infrastructure_error"), (REASON, "SandboxTimeoutException"), (OUTCOME, "dropped")): 1,
        ((CLASS, "infrastructure_error"), (REASON, "RuntimeError"), (OUTCOME, "dropped")): 1,
        ((CLASS, "judge_failed"), (OUTCOME, "dropped")): 1,
    }


def test_free_text_reasons_do_not_become_attributes(collected_metrics):  # noqa: F811
    gym_metrics.record_rollout_completed(
        "dropped", failure_class="agent_request_failed", failure_reason="Timeout on reading data from socket"
    )
    gym_metrics.record_rollout_completed("dropped", failure_class="agent_request_failed", failure_reason="x" * 200)
    assert _by_attrs(collected_metrics) == {((CLASS, "agent_request_failed"), (OUTCOME, "dropped")): 2}


def test_recording_without_telemetry_is_a_no_op(monkeypatch):
    from nemo_gym.telemetry import setup as telemetry_setup

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    gym_metrics.record_rollout_completed("scored")


def test_driver_outcome_helper_maps_verdicts(collected_metrics):  # noqa: F811
    rollout_collection._record_rollout_outcome({"reward": 1.0}, None)
    rollout_collection._record_rollout_outcome(
        {"_ng_failure_class": "infrastructure_error", "failure_reason": "VerifierTimeoutError"}, "infrastructure_error"
    )
    rollout_collection._record_rollout_outcome({rollout_collection.NG_NO_PERSIST_KEY: True}, None)

    assert _by_attrs(collected_metrics) == {
        ((OUTCOME, "scored"),): 1,
        ((CLASS, "infrastructure_error"), (REASON, "VerifierTimeoutError"), (OUTCOME, "dropped")): 1,
        ((CLASS, "no_persist"), (OUTCOME, "dropped")): 1,
    }
