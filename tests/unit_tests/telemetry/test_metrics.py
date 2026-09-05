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
"""Gym metric recording, against a real in-memory OTel metric reader.

These assert on the metrics that actually come out the other end, not on whether a
wrapper was called — the recording path runs through nemo-lens's `record_gym_metrics`,
where a wrong kwarg name would silently record nothing.
"""

import pytest

from nemo_gym.telemetry import metrics as telemetry_metrics
from nemo_gym.telemetry import setup as telemetry_setup
from tests.unit_tests.telemetry.conftest import requires_lens


#: These exercise the telemetry-enabled path, which needs nemo-lens. The absent-lens path
#: is covered by test_fallbacks.py, which runs either way.
pytestmark = requires_lens


@pytest.fixture
def collected_metrics(monkeypatch):
    """A live meter wired to an in-memory reader, installed as the process handle.

    Returns a callable that flushes and returns ``{metric_name: [data points]}``.
    """
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())

    def collect():
        data = reader.get_metrics_data()
        out = {}
        for resource_metric in data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    out[metric.name] = list(metric.data.data_points)
        return out

    return collect


# --------------------------------------------------------------------------- #
# Off path
# --------------------------------------------------------------------------- #


def test_recording_without_telemetry_is_a_no_op():
    """Every call site can record unconditionally; an uninitialised process ignores it."""
    assert telemetry_setup.get_telemetry() is None
    telemetry_metrics.record_rollout_duration(12.5)
    telemetry_metrics.record_verify(3.0, succeeded=True)
    telemetry_metrics.record_active_servers(4)


def test_recording_is_a_no_op_on_a_non_exporting_process(monkeypatch, collected_metrics):
    """A non-exporting rank holds a handle but must emit nothing."""

    class _Silent:
        is_exporting = False
        meter = None  # touching this would raise, which is the point

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Silent())
    telemetry_metrics.record_rollout_duration(12.5)
    telemetry_metrics.record_active_servers(2)


def test_recording_errors_never_reach_the_caller(monkeypatch):
    """Telemetry must not be able to fail a rollout."""

    class _Broken:
        is_exporting = True

        @property
        def meter(self):
            raise RuntimeError("meter is gone")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Broken())
    telemetry_metrics.record_rollout_duration(1.0)
    telemetry_metrics.record_verify(1.0, succeeded=False)


# --------------------------------------------------------------------------- #
# On path
# --------------------------------------------------------------------------- #


def test_rollout_duration_reaches_the_named_instrument(collected_metrics):
    telemetry_metrics.record_rollout_duration(125.0)
    metrics = collected_metrics()

    assert "gym.rollout.duration_ms" in metrics, f"got {sorted(metrics)}"
    point = metrics["gym.rollout.duration_ms"][0]
    assert point.count == 1
    assert point.sum == 125.0


def test_verify_records_duration_and_success_rate(collected_metrics):
    telemetry_metrics.record_verify(40.0, succeeded=True)
    metrics = collected_metrics()

    assert metrics["gym.verify.duration_ms"][0].sum == 40.0
    assert metrics["gym.verify.success_rate"][0].value == 1.0


def test_verify_success_rate_is_the_cumulative_fraction(collected_metrics):
    """The gauge is last-value, so it must carry a meaningful running figure.

    Three successes and one failure is 0.75 — not 0 (the last outcome) and not 1.0.
    """
    telemetry_metrics.record_verify(1.0, succeeded=True)
    telemetry_metrics.record_verify(1.0, succeeded=True)
    telemetry_metrics.record_verify(1.0, succeeded=True)
    telemetry_metrics.record_verify(1.0, succeeded=False)

    metrics = collected_metrics()
    assert metrics["gym.verify.success_rate"][0].value == pytest.approx(0.75)
    assert metrics["gym.verify.duration_ms"][0].count == 4


def test_verify_success_rate_starts_at_zero_after_a_failure(collected_metrics):
    telemetry_metrics.record_verify(1.0, succeeded=False)
    assert collected_metrics()["gym.verify.success_rate"][0].value == 0.0


def test_active_servers_gauge_is_set(collected_metrics):
    telemetry_metrics.record_active_servers(3)
    assert collected_metrics()["gym.servers.active"][0].value == 3


def test_active_servers_reflects_the_latest_value(collected_metrics):
    """Last-value semantics — which is exactly why only one process may write it."""
    telemetry_metrics.record_active_servers(3)
    telemetry_metrics.record_active_servers(5)
    assert collected_metrics()["gym.servers.active"][0].value == 5


def test_request_duration_instrument_is_deliberately_not_used(collected_metrics):
    """`gym.server.request_duration_ms` has no attributes, so it would merge every
    endpoint of every server type into one histogram. Gym uses the FastAPI
    instrumentor's dimensioned `http.server.*` instead; this pins that decision so a
    later change has to be deliberate."""
    telemetry_metrics.record_rollout_duration(1.0)
    telemetry_metrics.record_verify(1.0, succeeded=True)
    telemetry_metrics.record_active_servers(1)

    assert "gym.server.request_duration_ms" not in collected_metrics()
    assert not hasattr(telemetry_metrics, "record_server_request_duration")


def test_recorded_metric_names_are_a_subset_of_the_lens_gym_instruments(collected_metrics):
    """Everything Gym emits must be an instrument nemo-lens actually declares.

    Catches a typo'd kwarg, which `record_gym_metrics` would otherwise reject as an
    unexpected keyword — or worse, silently ignore if it grew a **kwargs.
    """
    telemetry_metrics.record_rollout_duration(1.0)
    telemetry_metrics.record_verify(1.0, succeeded=True)
    telemetry_metrics.record_active_servers(1)

    declared = {
        "gym.server.request_duration_ms",
        "gym.rollout.duration_ms",
        "gym.verify.duration_ms",
        "gym.verify.success_rate",
        "gym.servers.active",
    }
    assert set(collected_metrics()) <= declared
