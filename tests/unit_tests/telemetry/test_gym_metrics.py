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
"""``nemo_gym.telemetry.gym_metrics`` — the attributed instruments ``record_gym_metrics``
cannot express, created directly against the OTel Metrics API.

Mirrors ``test_metrics.py``'s approach: assert on data points that actually come out of a
real in-memory reader, not on whether an internal function was called.
"""

import pytest

from nemo_gym.telemetry import gym_metrics
from nemo_gym.telemetry import setup as telemetry_setup
from tests.unit_tests.telemetry.conftest import requires_lens


pytestmark = requires_lens


@pytest.fixture
def collected_metrics(monkeypatch):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    gym_metrics._reset_for_testing()

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
    assert telemetry_setup.get_telemetry() is None
    gym_metrics.record_queue_wait(1.0, site="rollout_driver")
    gym_metrics.record_rollout_completed(outcome="success")
    gym_metrics.record_sandbox_startup(1.0, provider="docker")
    gym_metrics.record_sandbox_create_retry(provider="daytona")
    gym_metrics.record_model_call_duration(1.0, dialect="responses", server_name="m")
    gym_metrics.record_model_ttft(1.0, dialect="responses", server_name="m")
    gym_metrics.record_http_timeout(internal=False)
    gym_metrics.record_retry(reason="timeout")


def test_recording_errors_never_reach_the_caller(monkeypatch):
    class _Broken:
        is_exporting = True

        @property
        def meter(self):
            raise RuntimeError("meter is gone")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Broken())
    gym_metrics.record_queue_wait(1.0, site="rollout_driver")
    gym_metrics.record_retry(reason="other")


# --------------------------------------------------------------------------- #
# On path
# --------------------------------------------------------------------------- #


def test_queue_wait_is_attributed_by_site(collected_metrics):
    gym_metrics.record_queue_wait(50.0, site="rollout_driver")
    gym_metrics.record_queue_wait(10.0, site="model.openai_model")

    points = collected_metrics()["gym.concurrency.queue_wait_duration_ms"]
    by_site = {dict(p.attributes)["nemo.gym.concurrency.site"]: p for p in points}
    assert by_site["rollout_driver"].sum == 50.0
    assert by_site["model.openai_model"].sum == 10.0


def test_rollout_completed_counts_by_outcome(collected_metrics):
    gym_metrics.record_rollout_completed(outcome="success")
    gym_metrics.record_rollout_completed(outcome="success")
    gym_metrics.record_rollout_completed(outcome="failure")

    points = collected_metrics()["gym.rollout.completed_total"]
    by_outcome = {dict(p.attributes)["nemo.gym.rollout.outcome"]: p.value for p in points}
    assert by_outcome["success"] == 2
    assert by_outcome["failure"] == 1


def test_sandbox_startup_is_attributed_by_provider(collected_metrics):
    gym_metrics.record_sandbox_startup(900.0, provider="daytona")
    point = collected_metrics()["gym.sandbox.startup_duration_ms"][0]
    assert point.sum == 900.0
    assert dict(point.attributes)["nemo.gym.sandbox.provider"] == "daytona"


def test_sandbox_create_retry_counts_by_provider(collected_metrics):
    gym_metrics.record_sandbox_create_retry(provider="daytona")
    gym_metrics.record_sandbox_create_retry(provider="daytona")
    point = collected_metrics()["gym.sandbox.create_retry_total"][0]
    assert point.value == 2
    assert dict(point.attributes)["nemo.gym.sandbox.provider"] == "daytona"


def test_model_call_duration_is_attributed_by_dialect_and_server(collected_metrics):
    gym_metrics.record_model_call_duration(75.0, dialect="responses", server_name="vllm_model")
    point = collected_metrics()["gym.model.call_duration_ms"][0]
    assert point.sum == 75.0
    attrs = dict(point.attributes)
    assert attrs["nemo.gym.model.dialect"] == "responses"
    assert attrs["nemo.gym.server.name"] == "vllm_model"


def test_model_ttft_is_a_distinct_instrument_from_call_duration(collected_metrics):
    gym_metrics.record_model_call_duration(75.0, dialect="responses", server_name="m")
    gym_metrics.record_model_ttft(20.0, dialect="responses", server_name="m")
    metrics = collected_metrics()
    assert metrics["gym.model.call_duration_ms"][0].sum == 75.0
    assert metrics["gym.model.ttft_ms"][0].sum == 20.0


def test_http_timeout_counts(collected_metrics):
    gym_metrics.record_http_timeout(internal=False)
    gym_metrics.record_http_timeout(internal=True)
    points = collected_metrics()["gym.http.timeout_total"]
    by_internal = {dict(p.attributes)["nemo.gym.http.internal"]: p.value for p in points}
    assert by_internal[False] == 1
    assert by_internal[True] == 1


def test_retry_counts_by_reason(collected_metrics):
    gym_metrics.record_retry(reason="timeout")
    gym_metrics.record_retry(reason="timeout")
    gym_metrics.record_retry(reason="server_disconnected")
    points = collected_metrics()["gym.http.retry_total"]
    by_reason = {dict(p.attributes)["nemo.gym.http.retry_reason"]: p.value for p in points}
    assert by_reason["timeout"] == 2
    assert by_reason["server_disconnected"] == 1


def test_instrument_cache_is_scoped_per_meter(collected_metrics, monkeypatch):
    """A second meter (e.g. a fresh provider installed by another test) must not reuse an
    instrument bound to the first meter's provider -- that raised in earlier nemo-lens
    behavior and, even where it doesn't raise, silently orphans data points."""
    gym_metrics.record_queue_wait(1.0, site="a")
    first = collected_metrics()
    assert "gym.concurrency.queue_wait_duration_ms" in first

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader2 = InMemoryMetricReader()
    provider2 = MeterProvider(metric_readers=[reader2])

    class _Handle2:
        is_exporting = True
        meter = provider2.get_meter("test2")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle2())
    gym_metrics.record_queue_wait(2.0, site="b")

    data = reader2.get_metrics_data()
    names = {m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics}
    assert "gym.concurrency.queue_wait_duration_ms" in names
