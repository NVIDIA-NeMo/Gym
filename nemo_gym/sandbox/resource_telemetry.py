# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-effort resource records in the existing sandbox trace stream."""

import asyncio
import logging
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxResourceAmounts, SandboxSpec
from nemo_gym.telemetry._fallbacks import is_span_group_enabled, managed_span, safe_set_span_attributes
from nemo_gym.telemetry.span_groups import GymSpanGroup


LOGGER = logging.getLogger(__name__)
PREFIX = "nemo.gym.sandbox."
Attribute = str | int | float | bool


@dataclass(frozen=True)
class ResourceDesignator:
    """Opaque infrastructure identity interpreted by kind, within an optional scope.

    This is not a network endpoint or reconnect descriptor and contains no credentials.
    Scope identifies the deployment/cluster/daemon in which the ID is meaningful.
    """

    kind: str
    id: str
    scope: str | None = None


@dataclass(frozen=True)
class ComponentAllocation:
    """Observed reservations and limits for one explicitly named component."""

    name: str
    requests: SandboxResourceAmounts = field(default_factory=SandboxResourceAmounts)
    limits: SandboxResourceAmounts = field(default_factory=SandboxResourceAmounts)


@dataclass(frozen=True)
class ResourceAllocation:
    """Provider-independent allocation record; placement is optional."""

    source: str
    components: tuple[ComponentAllocation, ...] = ()
    designator: ResourceDesignator | None = None
    host: ResourceDesignator | None = None


def amount_attributes(amounts: SandboxResourceAmounts, prefix: str) -> dict[str, Attribute]:
    """Serialize shared resource quantities without losing unspecified values."""
    return {PREFIX + prefix + name: value for name, value in asdict(amounts).items() if value is not None}


def allocation_attributes(allocation: ResourceAllocation) -> dict[str, Attribute]:
    """Use identical OTEL keys for every provider's typed allocation."""
    attributes: dict[str, Attribute] = {PREFIX + "allocation.source": allocation.source}
    for key in ("designator", "host"):
        reference = getattr(allocation, key)
        if reference is not None:
            for name, value in asdict(reference).items():
                if value is not None:
                    attributes[PREFIX + f"allocation.{key}.{name}"] = value
    attributes[PREFIX + "allocation.component_count"] = len(allocation.components)
    for index, component in enumerate(allocation.components):
        prefix = f"allocation.component.{index}."
        attributes[PREFIX + prefix + "name"] = component.name
        attributes.update(amount_attributes(component.requests, prefix + "requests."))
        attributes.update(amount_attributes(component.limits, prefix + "limits."))
    return attributes


@runtime_checkable
class ResourceProvider(Protocol):
    """Optional read-only capability; values carry explicit provider/source semantics."""

    async def resource_allocation(self, handle: SandboxHandle) -> ResourceAllocation: ...

    async def resource_sample(self, handle: SandboxHandle) -> dict[str, Attribute]: ...


def requested_attributes(spec: SandboxSpec) -> dict[str, Attribute]:
    """Declared shape only. Absent values are unknown, never zero or effective allocation."""
    attributes: dict[str, Attribute] = {PREFIX + "requested.count": 1}
    attributes.update(amount_attributes(spec.resources, "requested."))
    attributes[PREFIX + "requested.sidecar_count"] = len(spec.sidecars)
    for index, sidecar in enumerate(spec.sidecars):
        key = PREFIX + f"requested.sidecar.{index}."
        attributes[key + "name"] = sidecar.name
        attributes.update(amount_attributes(sidecar.resources, f"requested.sidecar.{index}."))
    return attributes


def record_request(spec: SandboxSpec, attributes: Mapping[str, Attribute]) -> dict[str, Attribute]:
    """Give one logical sandbox request an identity, even if provisioning never succeeds."""
    owned = {**attributes, PREFIX + "request.id": uuid.uuid4().hex}
    with managed_span(GymSpanGroup.SANDBOX, "gym.sandbox.request", **owned, **requested_attributes(spec)):
        pass
    return owned


def sampling_interval() -> float:
    """Return zero unless resource traces are enabled, including an actual exporter."""
    if not is_span_group_enabled(GymSpanGroup.SANDBOX):
        return 0
    from nemo_gym.telemetry.setup import is_trace_exporter_active

    if not is_trace_exporter_active():
        return 0
    try:
        value = float(os.getenv("NEMO_GYM_OTEL_SANDBOX_RESOURCE_SAMPLE_INTERVAL_SECONDS", "10"))
        return value if math.isfinite(value) and value > 0 else 0
    except ValueError:
        return 0


class ResourceSampler:
    """One owner samples a sandbox; borrowed connections do not start duplicate samplers."""

    def __init__(
        self, provider: ResourceProvider, handle: SandboxHandle, attributes: Mapping[str, Attribute], interval: float
    ) -> None:
        self.provider = provider
        self.handle = handle
        self.attributes = dict(attributes)
        self.interval = interval
        self.task: asyncio.Task[None] | None = None
        self.allocation_recorded = False

    async def record(self, kind: str, *, final: bool = False) -> None:
        # The telemetry boundary must not prevent creation, execution or deletion.
        try:
            with managed_span(
                GymSpanGroup.SANDBOX,
                "gym.sandbox.resource." + kind,
                **self.attributes,
                **{PREFIX + "resource.final": final},
            ) as span:
                try:
                    async with asyncio.timeout(2):
                        if kind == "allocation":
                            allocation = await self.provider.resource_allocation(self.handle)
                            values = allocation_attributes(allocation)
                        else:
                            values = await self.provider.resource_sample(self.handle)
                    safe_set_span_attributes(span, {PREFIX + "resource.status": "ok", **values})
                    if kind == "allocation":
                        self.allocation_recorded = True
                    else:
                        from nemo_gym.telemetry.gym_metrics import record_sandbox_resource_sample

                        record_sandbox_resource_sample(values, self.attributes)
                except Exception as exc:
                    safe_set_span_attributes(
                        span,
                        {
                            PREFIX + "resource.status": "unavailable",
                            PREFIX + "resource.error_type": type(exc).__name__,
                        },
                    )
        except Exception:
            LOGGER.debug("Could not export sandbox resource record", exc_info=True)

    async def run(self) -> None:
        await self.record("allocation")
        while True:
            await self.record("sample")
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        self.task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        if not self.allocation_recorded:
            await self.record("allocation", final=True)
        await self.record("sample", final=True)
