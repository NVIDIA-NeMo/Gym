# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-side sampling of a sandbox's measured CPU and memory over its lifetime.

The ``AsyncSandbox`` that created a sandbox samples it once after create and once before close. A measurement
whose peak is only a sampled lower bound (process-tree sampling) is also sampled periodically, every
``NEMO_GYM_SANDBOX_USAGE_SAMPLE_INTERVAL_SECONDS`` (default 5; 0 turns periodic sampling off).
``NEMO_GYM_SANDBOX_USAGE=0`` turns all sampling off.

This sits next to :class:`nemo_gym.sandbox.resource_telemetry.ResourceSampler`, which is a different mechanism:

- ``SandboxUsageTracker`` (here): providers implementing ``SupportsSandboxUsage`` (Docker, Enroot, Apptainer). It
  reads the runtime's lifetime accounting (cgroup counters or the sandbox's process tree), runs whether or not
  telemetry is configured, and returns the result as data: ``AsyncSandbox.resource_usage``, the ``gym.sandbox.stop``
  span attributes and the stopped-sandbox listeners that fill ``SandboxObservation``.
- ``ResourceSampler``: providers implementing ``ResourceProvider`` (OpenSandbox). It records the sandbox's allocation
  and live utilization gauges as OTel spans and metrics, every
  ``NEMO_GYM_OTEL_SANDBOX_RESOURCE_SAMPLE_INTERVAL_SECONDS``, only while a trace exporter is active; nothing is
  returned to the caller.
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
    SandboxUsageUnavailable,
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

    ``final_read`` says whether the read in ``stop()``, before close, measured this source and scope. Without it the
    totals are unknown: ``cpu_usage_usec_total``, ``cpu_usage_usec_delta`` and ``memory_peak_bytes`` are None (never
    the create-time reading), and ``unavailable_reason`` says why once the sandbox has stopped:
    ``final_read_timeout``, ``final_read_failed``, or a reason the provider gave (``not_cgroup_root``,
    ``sandbox_gone``).
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
    final_read: bool = False
    unavailable_reason: str | None = None


def sampling_enabled() -> bool:
    return os.getenv("NEMO_GYM_SANDBOX_USAGE", "1").strip().lower() not in ("0", "false", "off", "no")


def usage_sample_interval() -> float:
    try:
        value = float(os.getenv("NEMO_GYM_SANDBOX_USAGE_SAMPLE_INTERVAL_SECONDS", DEFAULT_SAMPLE_INTERVAL_S))
    except ValueError:
        return DEFAULT_SAMPLE_INTERVAL_S
    return value if math.isfinite(value) and value > 0 else 0.0


def summarize(
    samples: list[SandboxUsageSample], *, final_read: bool = True, unavailable_reason: str | None = None
) -> SandboxUsageSummary:
    """Fold the samples of one (source, scope) measurement, in the order taken.

    Without ``final_read`` the last sample is not a reading at close, so the totals are left unknown.
    """
    first, last = samples[0], samples[-1]
    peaks = [sample.memory_peak_bytes for sample in samples if sample.memory_peak_bytes is not None]
    delta = None
    if first.cpu_usage_usec_total is not None and last.cpu_usage_usec_total is not None and len(samples) > 1:
        delta = last.cpu_usage_usec_total - first.cpu_usage_usec_total
    if not final_read:
        return SandboxUsageSummary(
            source=last.source,
            scope=last.scope,
            clock=last.clock,
            first_sampled_at=first.sampled_at,
            last_sampled_at=last.sampled_at,
            sample_count=len(samples),
            cpu_kind=last.cpu_kind,
            peak_kind=next((sample.peak_kind for sample in reversed(samples) if sample.peak_kind), None),
            memory_current_bytes=last.memory_current_bytes,
            memory_anon_bytes=last.memory_anon_bytes,
            memory_file_bytes=last.memory_file_bytes,
            includes_page_cache=last.includes_page_cache,
            process_count=last.process_count,
            unavailable_reason=unavailable_reason,
        )
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
        final_read=True,
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
    """Samples one owned sandbox; a failed or slow read loses that sample, never the sandbox operation.

    Merging ``ResourceSampler`` into this class later (see the module docstring for how they differ): give
    OpenSandbox ``SupportsSandboxUsage`` (its container cgroup read through the provider's exec, or execd counters),
    let ``poll()`` also export each sample as the live OTel gauges ``ResourceSampler.record("sample")`` emits today
    when an exporter is active, and record the allocation once after create. ``ResourceSampler`` and
    ``ResourceProvider`` could then go. They stay separate for now because no provider implements both: the tracker
    reads lifetime counters on Docker, Enroot and Apptainer, while the sampler exports OpenSandbox's live gauges.
    """

    def __init__(self, provider: SupportsSandboxUsage, handle: SandboxHandle, *, interval: float) -> None:
        self.provider = provider
        self.handle = handle
        self.interval = interval
        self.samples: dict[tuple[str, str], list[SandboxUsageSample]] = {}
        self.first: asyncio.Task[tuple[SandboxUsageSample, ...]] | None = None
        self.task: asyncio.Task[None] | None = None
        # Why the latest read returned nothing, as an ``unavailable_reason``; None after a read that measured.
        self.last_failure: str | None = None
        # Set by stop(): the measurements its read covered, and why the others are missing.
        self.stopped = False
        self.final: set[tuple[str, str]] = set()
        self.final_failure: str | None = None

    async def sample(self) -> tuple[SandboxUsageSample, ...]:
        try:
            async with asyncio.timeout(SAMPLE_TIMEOUT_S):
                taken = tuple(await self.provider.usage(self.handle))
        except TimeoutError:
            LOGGER.debug("Sandbox usage sample timed out", exc_info=True)
            self.last_failure = "final_read_timeout"
            return ()
        except SandboxUsageUnavailable as exc:
            self.last_failure = exc.reason
            return ()
        except Exception:
            LOGGER.debug("Sandbox usage sample unavailable", exc_info=True)
            self.last_failure = "final_read_failed"
            return ()
        taken = tuple(sample for sample in taken if isinstance(sample, SandboxUsageSample))
        for sample in taken:
            self.samples.setdefault((sample.source, sample.scope), []).append(sample)
        self.last_failure = None if taken else "final_read_failed"
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
        first_timed_out = False
        if self.first is not None:
            pending = not self.first.done()
            await asyncio.gather(self.first, return_exceptions=True)
            self.first = None
            # A create read still running at stop() that then timed out means a hung runtime: a second read would
            # only add another timeout before close.
            first_timed_out = pending and self.last_failure == "final_read_timeout"
        taken = () if first_timed_out else await self.sample()
        self.stopped = True
        self.final = {(sample.source, sample.scope) for sample in taken}
        self.final_failure = self.last_failure or "final_read_failed"

    def summaries(self) -> tuple[SandboxUsageSummary, ...]:
        """One summary per measurement; before ``stop()`` none has a final read, so their totals are None."""
        return tuple(
            summarize(
                samples,
                final_read=key in self.final,
                unavailable_reason=self.final_failure if self.stopped and key not in self.final else None,
            )
            for key, samples in self.samples.items()
        )
