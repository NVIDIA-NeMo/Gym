# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import resource_telemetry as telemetry
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider
from nemo_gym.sandbox.providers.opensandbox.resource_telemetry import allocation_attributes
from tests.unit_tests.sandbox.test_sandbox import FakeSandboxProvider


P = telemetry.PREFIX
INSPECT = """Pod Name:       pod-1
Namespace:      test
Node:           node-1

Resources:
  sandbox:
    Requests: {'cpu': '100m', 'memory': '256Mi'}
    Limits:   {'cpu': '2', 'memory': '512Mi'}
  helper:
    Requests: {'cpu': '50m'}
    Limits:   {'cpu': '1'}
"""


def test_effective_resources_are_separate_from_declarations():
    requested = telemetry.requested_attributes(SandboxSpec(resources={"cpu": 2, "memory_mib": 512}))
    actual = allocation_attributes(INSPECT)
    assert requested[P + "requested.cpu"] == 2
    assert P + "requested.gpu" not in requested
    assert actual[P + "allocation.sandbox.requests.cpu"] == "100m"
    assert actual[P + "allocation.sandbox.limits.cpu"] == "2"
    assert actual[P + "allocation.helper.requests.cpu"] == "50m"
    assert actual[P + "allocation.node"] == "node-1"


@pytest.mark.parametrize("body", ["not supported", "Resources:\n", "\nResources:\n  sandbox:\n    Requests: {}"])
def test_missing_allocation_does_not_become_zero(body):
    with pytest.raises(ValueError):
        allocation_attributes(body)


def test_sidecar_shapes_remain_separate():
    values = telemetry.requested_attributes(
        SandboxSpec(resources={"cpu": 2}, sidecars=[{"name": "db", "image": "db", "resources": {"cpu": 1}}])
    )
    assert values[P + "requested.count"] == 1
    assert values[P + "requested.sidecar_count"] == 1
    assert values[P + "requested.sidecar.0.cpu"] == 1


@pytest.fixture
def records(monkeypatch):
    result = []

    @contextmanager
    def span(group, name, **attrs):
        result.append((name, attrs))
        yield attrs

    monkeypatch.setattr(telemetry, "managed_span", span)
    monkeypatch.setattr(telemetry, "safe_set_span_attributes", lambda span, attrs: span.update(attrs))
    return result


@pytest.mark.asyncio
async def test_sampler_preserves_identity_and_finishes_before_delete(records, monkeypatch):
    from nemo_gym.sandbox import api

    events = []

    class Provider(FakeSandboxProvider):
        async def resource_allocation(self, handle):
            events.append("allocation")
            return allocation_attributes(INSPECT)

        async def resource_sample(self, handle):
            events.append("sample")
            return {P + "sample.timestamp_ms": 1234, P + "sample.memory_used_mib": 7.5}

        async def close(self, handle):
            events.append("delete")
            await super().close(handle)

    monkeypatch.setattr(api, "sampling_interval", lambda: 100)
    sandbox = AsyncSandbox(Provider(), SandboxSpec())
    sandbox._resource_attributes = {"nemo.gym.rollout.id": "rollout-1"}
    # Keep the test independent of global Lens span configuration.
    monkeypatch.setattr(api, "is_span_group_enabled", lambda group: False)
    await sandbox.start()
    await asyncio.sleep(0)
    await sandbox.stop()
    assert events[-1] == "delete"
    assert events.count("allocation") == 1
    sample_records = [attrs for name, attrs in records if name.endswith(".sample")]
    assert sample_records[-1][P + "resource.final"] is True
    assert all(attrs["nemo.gym.rollout.id"] == "rollout-1" for attrs in sample_records)
    assert sample_records[-1][P + "sample.timestamp_ms"] == 1234
    assert sandbox._resource_sampler is None


@pytest.mark.asyncio
async def test_diagnostic_failure_keeps_sample_and_cleanup(records):
    provider = SimpleNamespace(
        resource_allocation=AsyncMock(side_effect=ValueError("unsupported response")),
        resource_sample=AsyncMock(return_value={P + "sample.memory_used_mib": 3}),
    )
    sampler = telemetry.ResourceSampler(provider, SandboxHandle("id", "test", None), {}, 100)
    sampler.start()
    await sampler.stop()  # Also covers immediate stop before the background task ran.
    assert records[0][1][P + "resource.status"] == "unavailable"
    assert records[0][1][P + "resource.error_type"] == "ValueError"
    assert records[-1][1][P + "resource.status"] == "ok"
    assert sampler.task is None


@pytest.mark.asyncio
async def test_stopping_cancels_inflight_sampling(records):
    entered = asyncio.Event()
    calls = 0

    async def sample(handle):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await asyncio.Event().wait()
        return {P + "sample.memory_used_mib": 1}

    provider = SimpleNamespace(resource_allocation=AsyncMock(return_value={}), resource_sample=sample)
    sampler = telemetry.ResourceSampler(provider, SandboxHandle("id", "test", None), {}, 100)
    sampler.start()
    await entered.wait()
    await asyncio.wait_for(sampler.stop(), 1)
    assert calls == 2
    assert sampler.task is None


@pytest.mark.asyncio
async def test_osb_snapshot_keeps_original_units_and_timestamp():
    raw = SimpleNamespace(
        get_metrics=AsyncMock(
            return_value=SimpleNamespace(
                timestamp=12345,
                cpu_count=2,
                cpu_used_percentage=12.5,
                memory_total_in_mib=512,
                memory_used_in_mib=7.5,
            )
        )
    )
    values = await OpenSandboxProvider().resource_sample(SandboxHandle("box", "opensandbox", raw))
    assert values[P + "sample.timestamp_ms"] == 12345
    assert values[P + "sample.cpu_used_percentage"] == 12.5
    assert values[P + "sample.memory_used_mib"] == 7.5


def test_sampling_requires_exporter_and_valid_interval(monkeypatch):
    from nemo_gym.telemetry import setup

    monkeypatch.setattr(telemetry, "is_span_group_enabled", lambda group: True)
    monkeypatch.setattr(setup, "is_trace_exporter_active", lambda: False)
    assert telemetry.sampling_interval() == 0
    monkeypatch.setattr(setup, "is_trace_exporter_active", lambda: True)
    for value in ("0", "nan", "bad", "-1", "inf"):
        monkeypatch.setenv("NEMO_GYM_OTEL_SANDBOX_RESOURCE_SAMPLE_INTERVAL_SECONDS", value)
        assert telemetry.sampling_interval() == 0
    monkeypatch.setenv("NEMO_GYM_OTEL_SANDBOX_RESOURCE_SAMPLE_INTERVAL_SECONDS", "5")
    assert telemetry.sampling_interval() == 5


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_inspection_uses_authenticated_control_plane_and_bounded_body(monkeypatch):
    pytest.importorskip("opensandbox", reason="requires nemo-gym[sandbox]")
    from nemo_gym.client import http_client

    reader = asyncio.StreamReader()
    reader.feed_data(INSPECT[:40].encode())
    reader.feed_data(INSPECT[40:].encode())
    reader.feed_eof()
    released = []
    response = SimpleNamespace(content=reader, raise_for_status=lambda: None, release=lambda: released.append(True))
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(http_client, "request", request)
    provider = OpenSandboxProvider(
        connection={
            "domain": "https://example.test/prefix",
            "api_key": "test-key",
            "tls_verify": True,
        }
    )
    attrs = await provider.resource_allocation(SandboxHandle("id/one", "opensandbox", None))
    assert attrs[P + "allocation.sandbox.requests.cpu"] == "100m"
    args, kwargs = request.call_args
    assert args == ("GET", "https://example.test/prefix/v1/sandboxes/id%2Fone/diagnostics/inspect")
    assert kwargs["headers"]["OPEN-SANDBOX-API-KEY"] == "test-key"
    assert kwargs["ssl"] is True
    assert kwargs["_max_num_tries"] == 1
    assert released == [True]
    await provider.aclose()
