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
"""Optional Prometheus sampling during rollout collection."""

import asyncio
import logging
import math
from contextlib import asynccontextmanager
from time import monotonic
from typing import AsyncIterator
from urllib.parse import urlencode, urlsplit, urlunsplit

from aiohttp import ClientTimeout
from prometheus_client.openmetrics.parser import text_string_to_metric_families as openmetrics_to_metric_families
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, Field, HttpUrl, model_validator

from nemo_gym.exporters import export_metrics, get_exporters
from nemo_gym.server_utils import request


logger = logging.getLogger(__name__)
PREFILL_INTERVAL_RATIOS = {
    "prefill_forwards_per_request": ("request_forwards", "completed_requests"),
    "prefill_tokens_per_request_forward": ("request_tokens", "request_forward_observations"),
    "prefill_tokens_per_forward": ("batch_tokens", "batches"),
    "prefill_requests_per_forward": ("batch_requests", "batches"),
    "prefill_underfilled_batch_fraction": ("underfilled_batches", "batches"),
}
PREFILL_READINESS_FIELDS = ("ready_requests", "ready_unscheduled_requests", "unexamined_requests")
PREFILL_OUTCOME_METRICS = {
    "prefill_requests_per_step": ("request_outcomes", "outcome_steps", "outcome"),
    "prefill_unexamined_requests_per_step": ("unexamined_by_stop_reason", "outcome_steps", "stop_reason"),
    "prefill_underfilled_requests_per_forward": (
        "underfilled_request_outcomes",
        "underfilled_outcome_steps",
        "outcome",
    ),
    "prefill_underfilled_unexamined_requests_per_forward": (
        "underfilled_unexamined_by_stop_reason",
        "underfilled_outcome_steps",
        "stop_reason",
    ),
}
PREFILL_ENGINE_COUNTERS = {f"prefill_{field}" for field in (*PREFILL_READINESS_FIELDS, "readiness_steps")} | {
    f"prefill_{field}"
    for numerator, denominator, _ in PREFILL_OUTCOME_METRICS.values()
    for field in (numerator, denominator)
}
PREFILL_DERIVED_METRICS = (
    set(PREFILL_INTERVAL_RATIOS)
    | {
        "prefill_batch_token_fill_ratio",
        "prefill_underfill_reason_fraction",
        "prefill_kv_lookup_latency_ms",
        "prefill_kv_lookup_pending_latency_ms",
        "prefill_kv_lookup_queue_latency_ms",
        "prefill_kv_lookup_execution_latency_ms",
        "prefill_kv_lookup_observation_latency_ms",
        "prefill_kv_lookup_client_latency_ms",
        "prefill_delayed_requests_per_step",
        "prefill_blocked_requests_per_step",
    }
    | {f"prefill_{field}_per_step" for field in PREFILL_READINESS_FIELDS}
    | set(PREFILL_OUTCOME_METRICS)
)
MOONCAKE_STORAGE_GAUGES = {"master_allocated_bytes", "master_total_capacity_bytes"}
MOONCAKE_EVICTION_GAUGES = {
    "master_eviction_memory_bytes",
    "master_eviction_blocked_bytes",
    "master_eviction_snapshot_timestamp_seconds",
    "master_offload_pending_bytes",
    "master_offload_pending_objects",
    "master_offload_oldest_age_seconds",
}
# Family names omit `_total`, including counters whose native names lack it.
MOONCAKE_EVICTION_COUNTERS = {
    f"master_{metric}{tier}"
    for metric in ("successful_evictions", "attempted_evictions", "evicted_key_count", "evicted_size_bytes")
    for tier in ("", "_mem", "_nof")
}
# These SGLang gauges describe ratios or averages, not additive quantities.
SGLANG_MEAN_GAUGES = {
    "token_usage",
    "full_token_usage",
    "swa_token_usage",
    "mamba_usage",
    "pending_prealloc_token_usage",
    "cache_hit_rate",
    "spec_accept_length",
    "spec_accept_rate",
    "spec_cap_length",
    "spec_block_accept_length",
    "utilization",
    "fwd_occupancy",
    "new_token_ratio",
    "lora_pool_utilization",
}


class InferenceMetricsConfig(BaseModel, extra="forbid"):
    """Opt-in sampling of vLLM, SGLang, router, and Mooncake Prometheus endpoints."""

    enabled: bool = False
    endpoints: dict[str, HttpUrl] = Field(default_factory=dict, description="Replica name to full /metrics URL.")
    router_endpoints: dict[str, HttpUrl] = Field(default_factory=dict, description="Router name to /metrics URL.")
    mooncake_endpoint: HttpUrl | None = Field(
        default=None, description="Single Mooncake master /metrics URL for storage usage and eviction metrics."
    )
    cache_inventory: bool = Field(default=False, description="Enable expensive debug cache inventory snapshots.")
    cache_inventory_interval_s: float = Field(default=10.0, ge=10, allow_inf_nan=False)
    cache_inventory_timeout_s: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    mooncake_tenant_id: str = "default"
    interval_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    timeout_s: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    metrics: list[str] | None = Field(
        default=None,
        description=(
            "Optional exact sample allowlist; by default export all vLLM, SGLang and router gauges, counters, "
            "and histogram samples plus Mooncake storage and eviction metrics, excluding creation timestamps."
        ),
    )

    @model_validator(mode="after")
    def validate_enabled(self) -> "InferenceMetricsConfig":
        if self.enabled and not (self.endpoints or self.router_endpoints or self.mooncake_endpoint):
            raise ValueError("Enabled inference_metrics requires endpoints")
        if self.endpoints.keys() & self.router_endpoints.keys():
            raise ValueError("Replica and router endpoint names must be distinct")
        names = [*self.endpoints, *self.router_endpoints]
        if self.mooncake_endpoint is not None and "mooncake" in names:
            raise ValueError("Endpoint name 'mooncake' is reserved when mooncake_endpoint is configured")
        if any(not name or not all(c.isalnum() or c in "_-" for c in name) for name in names):
            raise ValueError(
                "Inference metrics replica names must contain only letters, numbers, underscores or hyphens"
            )
        return self


class InferenceMetricsCollector:
    """Collect independent replica snapshots and publish through active exporters."""

    def __init__(self, config: InferenceMetricsConfig) -> None:
        self.config = config
        self.previous: dict[str, tuple[float, float]] = {}
        self.failed: set[str] = set()
        self.cache_layouts: dict[str, set[tuple[str, str, str]]] = {}
        self._next_inventory_at = 0.0
        self._inventory_task: asyncio.Task[None] | None = None

    def parse(self, replica: str, payload: str, sampled_at: float, *, content_type: str = "") -> dict[str, float]:
        """Aggregate labeled series and derive reset-aware counter and histogram rates."""
        result = {}
        cache_layout: set[tuple[str, str, str]] = set()
        gauge_counts = {}
        incomplete_rates = set()
        # OpenMetrics declares counters without the sample's `_total` suffix.
        # The legacy parser classifies those samples as unknown and drops rates.
        parser = (
            openmetrics_to_metric_families
            if content_type.split(";", 1)[0].strip().lower() == "application/openmetrics-text"
            else text_string_to_metric_families
        )
        for family in parser(payload):
            if family.type not in {"gauge", "counter", "histogram"}:
                continue
            for sample in family.samples:
                if sample.name == "vllm:mooncake_cache_layout":
                    if (
                        self.config.cache_inventory
                        and sample.value == 1
                        and all(sample.labels.get(k) for k in ("scope", "prefix", "kind"))
                    ):
                        cache_layout.add(tuple(sample.labels[k] for k in ("scope", "prefix", "kind")))
                    continue
                if sample.name == "vllm:cache_inventory" and not self.config.cache_inventory:
                    continue
                is_mooncake_master = replica == "mooncake" and (
                    (family.type == "gauge" and sample.name in MOONCAKE_STORAGE_GAUGES | MOONCAKE_EVICTION_GAUGES)
                    or (family.type == "counter" and family.name.removesuffix("_total") in MOONCAKE_EVICTION_COUNTERS)
                )
                if (
                    (not sample.name.startswith(("vllm:", "sglang:", "vllm_router_")) and not is_mooncake_master)
                    # Prometheus exposes creation timestamps as separate gauges.
                    or sample.name.endswith("_created")
                    or (self.config.metrics is not None and sample.name not in self.config.metrics)
                    or not math.isfinite(sample.value)
                ):
                    continue
                metric_replica = replica
                if is_mooncake_master:
                    # Master metrics are cluster totals; exclude per-segment duplicates.
                    namespace, name = "mooncake", sample.name.removeprefix("master_")
                    metric_replica = "total"
                elif sample.name.startswith("vllm_router_"):
                    namespace, name = "router", sample.name.removeprefix("vllm_router_")
                else:
                    namespace, name = sample.name.split(":", 1)
                labels = urlencode(sorted(sample.labels.items()))
                suffix = "".join(
                    f"/{label}/{value.strip('/')}"
                    for label, value in sorted(sample.labels.items())
                    if label != "model_name"
                    and (
                        label != "engine"
                        or name == "cache_inventory"
                        or name.startswith("dbo_")
                        or name.removesuffix("_total") in PREFILL_ENGINE_COUNTERS
                    )
                    # Role is already captured by replica names (prefill0/decode0).
                    # Keep rank, mode, priority and histogram labels distinct.
                    and not (namespace == "sglang" and label == "engine_type")
                )
                key = f"{namespace}/{metric_replica}/{name}{suffix}"
                series_key = f"{key}/{labels}"
                result[key] = result.get(key, 0.0) + sample.value
                if family.type == "gauge" and (
                    name == "kv_cache_usage_perc" or (namespace == "sglang" and name in SGLANG_MEAN_GAUGES)
                ):
                    gauge_counts[key] = gauge_counts.get(key, 0) + 1
                if family.type in {"counter", "histogram"}:
                    previous = self.previous.get(series_key)
                    self.previous[series_key] = (sampled_at, sample.value)
                    rate_key = f"{namespace}/{metric_replica}/{name.removesuffix('_total')}_per_second{suffix}"
                    if previous is not None:
                        previous_time, previous_value = previous
                        # A reset has an unknown start time: establish a fresh baseline.
                        if sampled_at > previous_time and sample.value >= previous_value:
                            result[rate_key] = result.get(rate_key, 0.0) + (sample.value - previous_value) / (
                                sampled_at - previous_time
                            )
                        else:
                            incomplete_rates.add(rate_key)
                    else:
                        incomplete_rates.add(rate_key)
        for key, count in gauge_counts.items():
            result[key] /= count
        for key in incomplete_rates:
            result.pop(key, None)
        if self.config.cache_inventory:
            self.cache_layouts[replica] = cache_layout
        self.add_derived_metrics(result, f"vllm/{replica}/")
        allocated = result.get("mooncake/total/allocated_bytes")
        capacity = result.get("mooncake/total/total_capacity_bytes")
        if allocated is not None and allocated >= 0 and capacity is not None and capacity > 0:
            result["mooncake/total/kv_cache_usage_perc"] = allocated / capacity
        return result

    @staticmethod
    def add_derived_metrics(metrics: dict[str, float], prefix: str) -> None:
        """Derive interval cache percentages and observation-weighted prefill averages."""
        overlap_prefix = f"{prefix}dbo_overlap_seconds_sum_per_second"
        overlaps = {
            key.removeprefix(overlap_prefix): value
            for key, value in metrics.items()
            if key == overlap_prefix or key.startswith(overlap_prefix + "/engine/")
        }
        for phase in ("comm", "compute"):
            pooled_overlap = pooled_duration = 0.0
            for suffix, overlap in overlaps.items():
                duration = metrics.get(f"{prefix}dbo_{phase}_seconds_sum_per_second{suffix}")
                if duration is not None and duration > 0:
                    metrics[f"{prefix}dbo_{phase}_overlap_fraction{suffix}"] = overlap / duration
                    pooled_overlap += overlap
                    pooled_duration += duration
            if pooled_duration > 0:
                metrics[f"{prefix}dbo_{phase}_overlap_fraction"] = pooled_overlap / pooled_duration
        # Preserve each DP engine's readiness counts and pool sums/counts for the
        # node summary. Missing/reset engines cannot contribute an interval rate.
        step_prefix = f"{prefix}prefill_readiness_steps_per_second"
        readiness_steps = {
            key.removeprefix(step_prefix): value
            for key, value in metrics.items()
            if key == step_prefix or key.startswith(step_prefix + "/engine/")
        }
        for field in PREFILL_READINESS_FIELDS:
            pooled_count = pooled_steps = 0.0
            for suffix, steps in readiness_steps.items():
                count = metrics.get(f"{prefix}prefill_{field}_per_second{suffix}")
                if count is not None and steps > 0:
                    metrics[f"{prefix}prefill_{field}_per_step{suffix}"] = count / steps
                    pooled_count += count
                    pooled_steps += steps
            if pooled_steps > 0:
                metrics[f"{prefix}prefill_{field}_per_step"] = pooled_count / pooled_steps
        for name, (numerator, denominator, label) in PREFILL_OUTCOME_METRICS.items():
            step_prefix = f"{prefix}prefill_{denominator}_per_second"
            steps_by_engine = {
                key.removeprefix(step_prefix): value
                for key, value in metrics.items()
                if key == step_prefix or key.startswith(step_prefix + "/engine/")
            }
            pooled: dict[str, tuple[float, float]] = {}
            for engine_suffix, steps in steps_by_engine.items():
                if steps <= 0:
                    continue
                count_prefix = f"{prefix}prefill_{numerator}_per_second{engine_suffix}/{label}/"
                for key, count in list(metrics.items()):
                    if not key.startswith(count_prefix):
                        continue
                    value = key.removeprefix(count_prefix)
                    metrics[f"{prefix}{name}{engine_suffix}/{label}/{value}"] = count / steps
                    total_count, total_steps = pooled.get(value, (0.0, 0.0))
                    pooled[value] = (total_count + count, total_steps + steps)
            for value, (count, steps) in pooled.items():
                metrics[f"{prefix}{name}/{label}/{value}"] = count / steps
        for name, (numerator, denominator) in PREFILL_INTERVAL_RATIOS.items():
            total = metrics.get(f"{prefix}prefill_{numerator}_per_second")
            count = metrics.get(f"{prefix}prefill_{denominator}_per_second")
            if total is not None and count is not None and count > 0:
                metrics[f"{prefix}{name}"] = total / count
        # Scheduler-observed latency includes time until the completion is polled.
        # Pending-only latency excludes lookups answered on their first call.
        for kind in ("kv_lookup", "kv_lookup_pending"):
            duration = metrics.get(f"{prefix}prefill_{kind}_duration_ns_per_second")
            completed = metrics.get(f"{prefix}prefill_{kind}_completed_per_second")
            if duration is not None and completed is not None and completed > 0:
                metrics[f"{prefix}prefill_{kind}_latency_ms"] = duration / completed / 1e6
        # Instrumented client attempts share a denominator, unlike the generic
        # scheduler metric (which also includes immediate skips/other connectors).
        completed = metrics.get(f"{prefix}prefill_kv_lookup_timed_completed_per_second")
        if completed is not None and completed > 0:
            durations = [
                metrics.get(f"{prefix}prefill_kv_lookup_{phase}_duration_ns_per_second")
                for phase in ("queue", "execution", "observation")
            ]
            for phase, duration in zip(("queue", "execution", "observation"), durations):
                if duration is not None:
                    metrics[f"{prefix}prefill_kv_lookup_{phase}_latency_ms"] = duration / completed / 1e6
            if all(duration is not None for duration in durations):
                metrics[f"{prefix}prefill_kv_lookup_client_latency_ms"] = (
                    sum(duration for duration in durations if duration is not None) / completed / 1e6
                )
        steps = metrics.get(f"{prefix}prefill_delay_steps_per_second")
        if steps is not None and steps > 0:
            for field in ("delayed_requests", "blocked_requests"):
                reason_prefix = f"{prefix}prefill_{field}_by_reason_per_second/reason/"
                for key, value in list(metrics.items()):
                    if key.startswith(reason_prefix):
                        reason = key.removeprefix(reason_prefix)
                        metrics[f"{prefix}prefill_{field}_per_step/reason/{reason}"] = value / steps
        tokens = metrics.get(f"{prefix}prefill_batch_tokens_per_second")
        unused = metrics.get(f"{prefix}prefill_unused_token_slots_per_second")
        if tokens is not None and unused is not None and tokens + unused > 0:
            metrics[f"{prefix}prefill_batch_token_fill_ratio"] = tokens / (tokens + unused)
        batches = metrics.get(f"{prefix}prefill_batches_per_second")
        if batches is not None and batches > 0:
            reason_prefix = f"{prefix}prefill_underfilled_batches_by_reason_per_second/reason/"
            for key, value in list(metrics.items()):
                if key.startswith(reason_prefix):
                    reason = key.removeprefix(reason_prefix)
                    metrics[f"{prefix}prefill_underfill_reason_fraction/reason/{reason}"] = value / batches
        for key, value in list(metrics.items()):
            query_prefix = f"{prefix}prefix_cache_queries_per_second"
            if key == query_prefix or key.startswith(query_prefix + "/"):
                suffix = key[len(query_prefix) :]
                hits = metrics.get(f"{prefix}prefix_cache_hits_per_second{suffix}")
                if hits is not None and value > 0:
                    metrics[f"{prefix}prefix_cache_hit_rate{suffix}"] = 100 * hits / value

    async def scrape(self, replica: str, url: HttpUrl) -> dict[str, float] | None:
        """Publish one replica sample; failures do not interrupt rollout collection."""
        try:
            # Bound the entire operation, including Gym's HTTP retry loop.
            async with asyncio.timeout(self.config.timeout_s):
                response = await request(
                    "GET", str(url), _max_connection_retries=0, timeout=ClientTimeout(total=self.config.timeout_s)
                )
                async with response:
                    response.raise_for_status()
                    payload = await response.text()
                    content_type = response.headers.get("Content-Type", "")
            sampled_at = monotonic()
            metrics = self.parse(replica, payload, sampled_at, content_type=content_type)
            if not metrics:
                raise ValueError("No matching gauge/counter/histogram samples in metrics response")
            export_metrics(metrics)
            if replica in self.failed:
                logger.info("Inference metrics scraping recovered for replica %s", replica)
                self.failed.remove(replica)
            return metrics
        except Exception as exc:
            self.cache_layouts.pop(replica, None)
            if replica not in self.failed:
                # Avoid logging endpoint URLs, which may contain credentials.
                logger.warning(
                    "Inference metrics scrape failed for replica %s (%s); continuing", replica, type(exc).__name__
                )
                self.failed.add(replica)
            return None

    async def report_cache_inventory(self) -> None:
        """Ask the master for logical objects, using the expected worker namespaces.

        This observes metadata only: no Get calls, leases, or recency changes.
        Complete Mamba prefixes do not imply complete attention coverage.
        """
        if not self.config.cache_inventory:
            return
        namespaces = sorted(set().union(*self.cache_layouts.values())) if self.cache_layouts else []
        if self.config.mooncake_endpoint is None or not namespaces:
            return
        parts = urlsplit(str(self.config.mooncake_endpoint))
        path = parts.path.removesuffix("/metrics").rstrip("/") + "/cache_inventory"
        url = urlunsplit(parts._replace(path=path, query="", fragment=""))
        payload = {
            "tenant_id": self.config.mooncake_tenant_id,
            "namespaces": [{"scope": scope, "prefix": prefix, "kind": kind} for scope, prefix, kind in namespaces],
        }
        try:
            async with asyncio.timeout(self.config.cache_inventory_timeout_s):
                response = await request(
                    "POST",
                    url,
                    json=payload,
                    _max_connection_retries=0,
                    timeout=ClientTimeout(total=self.config.cache_inventory_timeout_s),
                )
                async with response:
                    response.raise_for_status()
                    snapshot = await response.json()
            values = dict(snapshot["counts"])
            values["snapshot_timestamp_seconds"] = snapshot["snapshot_timestamp_seconds"]
            values["scan_seconds"] = snapshot["scan_seconds"]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values.values()):
                raise ValueError("Invalid cache inventory counts")
            export_metrics(
                {f"mooncake/total/cache_inventory/{key.replace('.', '/')}": value for key, value in values.items()}
            )
            logger.info("Mooncake cache inventory (logical objects; Mamba completeness only): %s", values)
            self.failed.discard("cache_inventory")
        except Exception as exc:
            if "cache_inventory" not in self.failed:
                logger.warning(
                    "Mooncake cache inventory unavailable (%s); continuing (requires a master with /cache_inventory)",
                    type(exc).__name__,
                )
                self.failed.add("cache_inventory")

    async def run(self, stop: asyncio.Event) -> None:
        """Sample immediately, then periodically until the rollout scope closes."""
        try:
            await self._sample_loop(stop)
        finally:
            if self._inventory_task is not None:
                self._inventory_task.cancel()
                await asyncio.gather(self._inventory_task, return_exceptions=True)
                self._inventory_task = None

    async def _sample_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            endpoints = [*self.config.endpoints.items(), *self.config.router_endpoints.items()]
            if self.config.mooncake_endpoint is not None:
                endpoints.append(("mooncake", self.config.mooncake_endpoint))
            snapshots = await asyncio.gather(
                *(self.scrape(name, url) for name, url in endpoints),
            )
            if (
                self.config.cache_inventory
                and self.config.mooncake_endpoint is not None
                and any(self.cache_layouts.values())
                and (self._inventory_task is None or self._inventory_task.done())
                and monotonic() >= self._next_inventory_at
            ):
                self._next_inventory_at = monotonic() + self.config.cache_inventory_interval_s
                self._inventory_task = asyncio.create_task(self.report_cache_inventory())
            aggregates = {}
            for namespace in ("vllm", "sglang"):
                # Match metric and label paths; missing replicas/series are not zeros.
                replica_metrics = [
                    {
                        key.removeprefix(f"{namespace}/{replica}/"): value
                        for key, value in (snapshot or {}).items()
                        if key.startswith(f"{namespace}/{replica}/")
                    }
                    for replica, snapshot in zip(self.config.endpoints, snapshots)
                ]
                shared_metrics = (
                    set.intersection(*(set(metrics) for metrics in replica_metrics)) if replica_metrics else set()
                )
                for metric in sorted(shared_metrics):
                    values = [metrics[metric] for metrics in replica_metrics]
                    total = sum(values)
                    name = metric.split("/", 1)[0]
                    if namespace == "vllm" and (
                        name in PREFILL_DERIVED_METRICS
                        or name in {"dbo_comm_overlap_fraction", "dbo_compute_overlap_fraction"}
                    ):
                        continue  # Recompute from pooled sums/counts; never sum or average ratios.
                    if name == "cache_inventory":
                        continue  # Keep per-engine inventories; hashes are not globally additive.
                    nonadditive = namespace == "sglang" and name in SGLANG_MEAN_GAUGES
                    if name != "prefix_cache_hit_rate" and not nonadditive:
                        aggregates[f"{namespace}/total/{metric}"] = total
                    aggregates[f"{namespace}/mean/{metric}"] = total / len(replica_metrics)
                    if name in {"kv_cache_usage_perc", "num_requests_waiting"} or (
                        namespace == "sglang" and (nonadditive or name.endswith("queue_reqs"))
                    ):
                        aggregates[f"{namespace}/max/{metric}"] = max(values)
                    if name in {"prefix_cache_hit_rate", "cache_hit_rate"}:
                        aggregates[f"{namespace}/min/{metric}"] = min(values)
            self.add_derived_metrics(aggregates, "vllm/total/")
            if aggregates:
                export_metrics(aggregates)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.config.interval_s)
            except TimeoutError:
                pass


@asynccontextmanager
async def collect_inference_metrics(config: InferenceMetricsConfig) -> AsyncIterator[None]:
    """Own the sampler task and wait for its bounded final publication on exit."""
    if not config.enabled:
        yield
        return
    exporters = get_exporters()
    if not exporters:
        logger.warning("Inference metrics enabled without an active exporter; skipping collection")
        yield
        return
    exporter_names = ", ".join(type(exporter).__name__ for exporter in exporters)
    print(
        f"Inference metrics collection enabled: {len(config.endpoints)} replicas, "
        f"{len(config.router_endpoints)} routers, "
        f"{int(config.mooncake_endpoint is not None)} Mooncake masters, "
        f"interval={config.interval_s:g}s, timeout={config.timeout_s:g}s, "
        f"exporters={exporter_names}",
        flush=True,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(InferenceMetricsCollector(config).run(stop), name="inference-metrics")
    try:
        yield
    finally:
        stop.set()
        await task
