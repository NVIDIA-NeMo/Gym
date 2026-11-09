# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import resource_telemetry as telemetry
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxResourceAmounts, SandboxResources, SandboxSpec
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider
from nemo_gym.sandbox.providers.opensandbox.resource_telemetry import parse_allocation, resource_amounts
from nemo_gym.telemetry import _fallbacks as telemetry_fallbacks
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
    actual = telemetry.allocation_attributes(parse_allocation(INSPECT, scope="cluster-a"))
    assert actual[P + "allocation.designator.kind"] == "kubernetes"
    assert actual[P + "allocation.designator.scope"] == "cluster-a"
    assert actual[P + "allocation.designator.id"] == "namespaces/test/pods/pod-1"
    assert actual[P + "allocation.component.1.name"] == "sandbox"
    assert requested[P + "requested.cpu"] == 2
    assert P + "requested.gpu" not in requested
    assert actual[P + "allocation.component.1.requests.cpu"] == 0.1
    assert actual[P + "allocation.component.1.limits.cpu"] == 2
    assert actual[P + "allocation.component.0.requests.cpu"] == 0.05
    assert actual[P + "allocation.host.id"] == "node-1"


@pytest.mark.parametrize("body", ["not supported", "Resources:\n", "\nResources:\n  sandbox:\n    Requests: {}"])
def test_missing_allocation_does_not_become_zero(body):
    with pytest.raises(ValueError):
        parse_allocation(body)


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
            return parse_allocation(INSPECT)

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
    monkeypatch.setattr(telemetry_fallbacks, "is_span_group_enabled", lambda group: False)
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

    provider = SimpleNamespace(
        resource_allocation=AsyncMock(return_value=telemetry.ResourceAllocation(source="test")), resource_sample=sample
    )
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
    attrs = telemetry.allocation_attributes(
        await provider.resource_allocation(SandboxHandle("id/one", "opensandbox", None))
    )
    assert attrs[P + "allocation.component.1.requests.cpu"] == 0.1
    args, kwargs = request.call_args
    assert args == ("GET", "https://example.test/prefix/v1/sandboxes/id%2Fone/diagnostics/inspect")
    assert kwargs["headers"]["OPEN-SANDBOX-API-KEY"] == "test-key"
    assert kwargs["ssl"] is True
    assert kwargs["_max_num_tries"] == 1
    assert released == [True]
    await provider.aclose()


@pytest.mark.parametrize(
    "cpu, memory, disk, expected",
    [
        ("100m", "1536Ki", "512Mi", (0.1, 1.5, 0.5)),
        ("1e-3", "1M", "1G", (0.001, 1_000_000 / 2**20, 1_000_000_000 / 2**30)),
        ("1000000n", "1.5Gi", "2Gi", (0.001, 1536, 2)),
    ],
)
def test_resource_units_preserve_fractional_quantities(cpu, memory, disk, expected):
    amounts = resource_amounts({"cpu": cpu, "memory": memory, "ephemeral-storage": disk, "nvidia.com/gpu": "2"})
    assert (amounts.cpu, amounts.memory_mib, amounts.disk_gib) == expected
    assert amounts.gpu == 2
    assert amounts.gpu_type is None


@pytest.mark.parametrize("value", ["bad", "nan", "inf", "-1", True, "1e9999"])
def test_invalid_quantities_are_not_exported(value):
    with pytest.raises(ValueError):
        resource_amounts({"cpu": value})


def test_gpu_aliases_and_missing_resources():
    assert resource_amounts({"gpu": "0", "gpu_type": "a100"}).gpu == 0
    assert resource_amounts({"vendor.example/thing": "3"}) == SandboxResourceAmounts()
    for values in ({"gpu": "0.5"}, {"gpu": "1", "nvidia.com/gpu": "2"}):
        with pytest.raises(ValueError):
            resource_amounts(values)


def test_request_type_and_clamping_remain_compatible():
    from nemo_gym.sandbox.providers.base import clamp_resource_requests

    request = SandboxResources.from_mapping({"cpu": "0.5", "memory_mib": "512", "disk_gib": "1"})
    assert isinstance(request, SandboxResourceAmounts)
    assert request.memory_mib == 512 and isinstance(request.memory_mib, int)
    assert clamp_resource_requests(request, SandboxResources(cpu=0.25))[0]["cpu"] == 0.25


@pytest.mark.parametrize(
    "kind, scope, identity",
    [
        ("kubernetes", "cluster-a", "namespaces/ns/pods/pod"),
        ("docker-container", "daemon-a", "container-1"),
        ("ecs-task", "cluster-arn", "task-arn"),
        ("provider-sandbox", "deployment-a", "sandbox-1"),
    ],
)
def test_allocation_schema_is_identical_for_different_designators(kind, scope, identity):
    allocation = telemetry.ResourceAllocation(
        source="test",
        designator=telemetry.ResourceDesignator(kind=kind, scope=scope, id=identity),
        components=(telemetry.ComponentAllocation(name="main", limits=SandboxResourceAmounts(cpu=2)),),
    )
    attrs = telemetry.allocation_attributes(allocation)
    assert attrs == {
        P + "allocation.source": "test",
        P + "allocation.designator.kind": kind,
        P + "allocation.designator.scope": scope,
        P + "allocation.designator.id": identity,
        P + "allocation.component_count": 1,
        P + "allocation.component.0.name": "main",
        P + "allocation.component.0.limits.cpu": 2,
    }


def test_missing_placement_is_not_fabricated():
    allocation = parse_allocation(INSPECT[INSPECT.index("\nResources:") :])
    assert allocation.designator is None and allocation.host is None
    assert allocation.components[1].requests.cpu == 0.1
