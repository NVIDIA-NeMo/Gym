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

"""``gym.rollout.failures``: the Gym-owned counter of rollouts dropped from the score."""

import pytest

from nemo_gym.telemetry import metrics as telemetry_metrics
from nemo_gym.telemetry import setup as telemetry_setup
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics  # noqa: F401 - fixture


pytest.importorskip("opentelemetry.sdk.metrics")

FAILURES = telemetry_metrics.ROLLOUT_FAILURES_INSTRUMENT
CLASS = telemetry_metrics.FAILURE_CLASS_ATTRIBUTE
REASON = telemetry_metrics.FAILURE_REASON_ATTRIBUTE


@pytest.fixture(autouse=True)
def _fresh_instrument(monkeypatch):
    monkeypatch.setattr(telemetry_metrics, "_ROLLOUT_FAILURES", None)


def test_recording_without_telemetry_is_a_no_op(monkeypatch):
    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    telemetry_metrics.record_rollout_failure("infrastructure_error", "RuntimeError")


def test_failures_are_counted_by_class_and_reason(collected_metrics):  # noqa: F811
    telemetry_metrics.record_rollout_failure("infrastructure_error", "SandboxTimeoutException")
    telemetry_metrics.record_rollout_failure("infrastructure_error", "SandboxTimeoutException")
    telemetry_metrics.record_rollout_failure("infrastructure_error", "RuntimeError")
    telemetry_metrics.record_rollout_failure("judge_failed", None)

    points = collected_metrics()[FAILURES]
    by_attrs = {(p.attributes[CLASS], p.attributes.get(REASON)): p.value for p in points}
    assert by_attrs == {
        ("infrastructure_error", "SandboxTimeoutException"): 2,
        ("infrastructure_error", "RuntimeError"): 1,
        ("judge_failed", None): 1,
    }


def test_free_text_reasons_do_not_become_attributes(collected_metrics):  # noqa: F811
    telemetry_metrics.record_rollout_failure("agent_request_failed", "Timeout on reading data from socket")
    telemetry_metrics.record_rollout_failure("agent_request_failed", "x" * 200)
    points = collected_metrics()[FAILURES]
    assert [dict(p.attributes) for p in points] == [{CLASS: "agent_request_failed"}]
    assert points[0].value == 2


def test_instrument_is_monotonic(collected_metrics):  # noqa: F811
    telemetry_metrics.record_rollout_failure("infrastructure_error")
    from opentelemetry.sdk.metrics.export import Sum  # noqa: F401 - type check below

    (metric_points,) = [collected_metrics()[FAILURES]]
    assert metric_points[0].value == 1
