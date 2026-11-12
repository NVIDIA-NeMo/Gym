# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.telemetry import gym_metrics, setup
from tests.unit_tests.telemetry.conftest import requires_lens
from tests.unit_tests.telemetry.in_memory_meter import collected_metrics  # noqa: F401
from tests.unit_tests.telemetry.test_call_sites import recorded_spans  # noqa: F401


def test_samples_export_gauges_with_owner_but_not_timestamp_labels(collected_metrics, monkeypatch):  # noqa: F811
    monkeypatch.setattr(setup, "is_metrics_exporter_active", lambda: True)
    owner = {"nemo.gym.rollout.id": "rollout-1", "nemo.gym.sandbox.id": "box-1"}
    gym_metrics.record_sandbox_resource_sample(
        {
            "nemo.gym.sandbox.sample.timestamp_ms": 1234,
            "nemo.gym.sandbox.sample.cpu_used_percentage": 12.5,
            "nemo.gym.sandbox.sample.memory_used_mib": 4,
        },
        owner,
    )
    points = collected_metrics()
    assert points["gym.sandbox.cpu_used_percentage"][0].value == 12.5
    assert points["gym.sandbox.memory_used_mib"][0].attributes == owner
    assert "gym.sandbox.cpu_count" not in points


def test_disabled_metrics_export_nothing(collected_metrics, monkeypatch):  # noqa: F811
    monkeypatch.setattr(setup, "is_metrics_exporter_active", lambda: False)
    gym_metrics.record_sandbox_resource_sample({"nemo.gym.sandbox.sample.cpu_count": 2}, {})
    assert collected_metrics() == {}


@requires_lens
async def test_failed_create_preserves_requested_shape_and_identity(recorded_spans):  # noqa: F811
    import pytest

    from nemo_gym.sandbox.api import AsyncSandbox, SandboxSpec
    from tests.unit_tests.sandbox.test_sandbox import FakeSandboxProvider

    class Refusing(FakeSandboxProvider):
        async def create(self, spec):
            raise RuntimeError("no capacity")

    with pytest.raises(RuntimeError, match="no capacity"):
        await AsyncSandbox(Refusing(), SandboxSpec(resources={"cpu": 2, "memory_mib": 512})).start(
            span_attributes={"nemo.gym.rollout.id": "rollout-1"}
        )
    spans = recorded_spans()
    requested = next(span for span in spans if span.name == "gym.sandbox.request")
    started = next(span for span in spans if span.name == "gym.sandbox.start")
    assert requested.attributes["nemo.gym.sandbox.requested.cpu"] == 2
    assert requested.attributes["nemo.gym.sandbox.requested.memory_mib"] == 512
    assert requested.attributes["nemo.gym.rollout.id"] == "rollout-1"
    assert requested.attributes["nemo.gym.sandbox.request.id"] == started.attributes["nemo.gym.sandbox.request.id"]
    assert "nemo.gym.sandbox.id" not in started.attributes
