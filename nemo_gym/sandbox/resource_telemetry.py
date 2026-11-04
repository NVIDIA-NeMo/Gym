# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-effort resource records in the existing sandbox trace stream."""

import asyncio
import logging
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from typing import Protocol, runtime_checkable

from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec
from nemo_gym.telemetry._fallbacks import is_span_group_enabled, managed_span, safe_set_span_attributes
from nemo_gym.telemetry.span_groups import GymSpanGroup


LOGGER = logging.getLogger(__name__)
PREFIX = "nemo.gym.sandbox."
Attribute = str | int | float | bool


@runtime_checkable
class ResourceProvider(Protocol):
    """Optional read-only capability; values carry explicit provider/source semantics."""

    async def resource_allocation(self, handle: SandboxHandle) -> dict[str, Attribute]: ...

    async def resource_sample(self, handle: SandboxHandle) -> dict[str, Attribute]: ...


def requested_attributes(spec: SandboxSpec) -> dict[str, Attribute]:
    """Declared shape only. Absent values are unknown, never zero or effective allocation."""
    attributes: dict[str, Attribute] = {PREFIX + "requested.count": 1}
    for name, value in asdict(spec.resources).items():
        if value is not None:
            attributes[PREFIX + "requested." + name] = value
    attributes[PREFIX + "requested.sidecar_count"] = len(spec.sidecars)
    for index, sidecar in enumerate(spec.sidecars):
        key = PREFIX + f"requested.sidecar.{index}."
        attributes[key + "name"] = sidecar.name
        for name, value in asdict(sidecar.resources).items():
            if value is not None:
                attributes[key + name] = value
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
                        values = await (
                            self.provider.resource_allocation(self.handle)
                            if kind == "allocation"
                            else self.provider.resource_sample(self.handle)
                        )
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
