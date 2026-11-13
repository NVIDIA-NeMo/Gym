# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-side sampling of a sandbox's measured CPU and memory over its lifetime.

The ``AsyncSandbox`` that created a sandbox samples it once after create and once before close. A measurement
whose peak is only a sampled lower bound (process-tree sampling) is also sampled periodically, every
``NEMO_GYM_SANDBOX_USAGE_SAMPLE_INTERVAL_SECONDS`` (default 5; 0 turns periodic sampling off).
``NEMO_GYM_SANDBOX_USAGE=0`` turns all sampling off.
"""

import asyncio
import logging
import math
import os
from dataclasses import asdict, dataclass

from nemo_gym.sandbox.providers.base import (
    SandboxHandle,
    SandboxUsageKind,
    SandboxUsageSample,
    SandboxUsageScope,
    SupportsSandboxUsage,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_SAMPLE_INTERVAL_S = 5.0
SAMPLE_TIMEOUT_S = 10.0  # bounds every read, so a hung runtime delays stop() by at most this


@dataclass(frozen=True)
class SandboxUsageSummary:
    """One measurement of a sandbox over the samples its owner took.

    ``cpu_usage_usec_total`` is the last cumulative reading: the sandbox's CPU since it started for ``container``
    and ``process_tree``, and the whole shared cgroup's for ``shared_step``, where ``cpu_usage_usec_delta`` (last
    minus first reading) is the CPU everything in that cgroup used while this sandbox lived. ``memory_peak_bytes``
    is the largest peak read. Current, anonymous and file memory are from the last sample.
    """

    source: str
    scope: SandboxUsageScope
    clock: str
    first_sampled_at: float
    last_sampled_at: float
    sample_count: int
    cpu_usage_usec_total: int | None = None
    cpu_usage_usec_delta: int | None = None
    cpu_kind: SandboxUsageKind | None = None
    memory_peak_bytes: int | None = None
    peak_kind: SandboxUsageKind | None = None
    memory_current_bytes: int | None = None
    memory_anon_bytes: int | None = None
    memory_file_bytes: int | None = None
    includes_page_cache: bool | None = None
    process_count: int | None = None


def sampling_enabled() -> bool:
    return os.getenv("NEMO_GYM_SANDBOX_USAGE", "1").strip().lower() not in ("0", "false", "off", "no")


def sample_interval() -> float:
    try:
        value = float(os.getenv("NEMO_GYM_SANDBOX_USAGE_SAMPLE_INTERVAL_SECONDS", DEFAULT_SAMPLE_INTERVAL_S))
    except ValueError:
        return DEFAULT_SAMPLE_INTERVAL_S
    return value if math.isfinite(value) and value > 0 else 0.0


def summarize(samples: list[SandboxUsageSample]) -> SandboxUsageSummary:
    """Fold the samples of one (source, scope) measurement, in the order taken."""
    first, last = samples[0], samples[-1]
    peaks = [sample.memory_peak_bytes for sample in samples if sample.memory_peak_bytes is not None]
    delta = None
    if first.cpu_usage_usec_total is not None and last.cpu_usage_usec_total is not None and len(samples) > 1:
        delta = last.cpu_usage_usec_total - first.cpu_usage_usec_total
    return SandboxUsageSummary(
        source=last.source,
        scope=last.scope,
        clock=last.clock,
        first_sampled_at=first.sampled_at,
        last_sampled_at=last.sampled_at,
        sample_count=len(samples),
        cpu_usage_usec_total=last.cpu_usage_usec_total,
        cpu_usage_usec_delta=delta,
        cpu_kind=last.cpu_kind,
        memory_peak_bytes=max(peaks) if peaks else None,
        peak_kind=next((sample.peak_kind for sample in reversed(samples) if sample.peak_kind), None),
        memory_current_bytes=last.memory_current_bytes,
        memory_anon_bytes=last.memory_anon_bytes,
        memory_file_bytes=last.memory_file_bytes,
        includes_page_cache=last.includes_page_cache,
        process_count=last.process_count,
    )


def usage_span_attributes(summaries: tuple[SandboxUsageSummary, ...]) -> dict[str, str | int | float | bool]:
    """Span attributes for a stopped sandbox's usage: ``nemo.gym.sandbox.usage.<scope>.<source>.<field>``.

    Keyed by scope and source, since one scope can have several sources (Enroot's process tree and its
    completed execs); unmeasured values are left out.
    """
    attributes: dict[str, str | int | float | bool] = {}
    for summary in summaries:
        prefix = f"nemo.gym.sandbox.usage.{summary.scope}.{summary.source}."
        for name, value in asdict(summary).items():
            if value is not None and name not in ("scope", "source"):
                attributes[prefix + name] = value
    return attributes


class SandboxUsageTracker:
    """Samples one owned sandbox; a failed or slow read loses that sample, never the sandbox operation."""

    def __init__(self, provider: SupportsSandboxUsage, handle: SandboxHandle, *, interval: float) -> None:
        self.provider = provider
        self.handle = handle
        self.interval = interval
        self.samples: dict[tuple[str, str], list[SandboxUsageSample]] = {}
        self.first: asyncio.Task[tuple[SandboxUsageSample, ...]] | None = None
        self.task: asyncio.Task[None] | None = None

    async def sample(self) -> tuple[SandboxUsageSample, ...]:
        try:
            async with asyncio.timeout(SAMPLE_TIMEOUT_S):
                taken = tuple(await self.provider.usage(self.handle))
        except Exception:
            LOGGER.debug("Sandbox usage sample unavailable", exc_info=True)
            return ()
        for sample in taken:
            if isinstance(sample, SandboxUsageSample):
                self.samples.setdefault((sample.source, sample.scope), []).append(sample)
        return taken

    async def poll(self, first: "asyncio.Task[tuple[SandboxUsageSample, ...]]") -> None:
        # Shielded: stopping the poll must not cancel the create sample, the baseline of every delta.
        taken = await asyncio.shield(first)
        if not self.interval:
            return
        # An unavailable first read says nothing about the provider: retry it before deciding whether to poll.
        while not taken:
            await asyncio.sleep(self.interval)
            taken = await self.sample()
        if not any(sample.peak_kind == "sampled_lower_bound" for sample in taken):
            return
        while True:
            await asyncio.sleep(self.interval)
            await self.sample()

    def start(self) -> None:
        self.first = asyncio.create_task(self.sample())
        self.task = asyncio.create_task(self.poll(self.first))

    async def stop(self) -> None:
        """Take the sample before close; call while the sandbox still runs."""
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        if self.first is not None:
            await asyncio.gather(self.first, return_exceptions=True)
            self.first = None
        await self.sample()

    def summaries(self) -> tuple[SandboxUsageSummary, ...]:
        return tuple(summarize(samples) for samples in self.samples.values())
