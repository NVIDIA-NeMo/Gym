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
from urllib.parse import urlencode

from aiohttp import ClientTimeout
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, Field, HttpUrl, model_validator

from nemo_gym.exporters import export_metrics, get_exporters
from nemo_gym.server_utils import request


logger = logging.getLogger(__name__)


class InferenceMetricsConfig(BaseModel, extra="forbid"):
    """Opt-in sampling of vLLM and router Prometheus endpoints."""

    enabled: bool = False
    endpoints: dict[str, HttpUrl] = Field(default_factory=dict, description="Replica name to full /metrics URL.")
    router_endpoints: dict[str, HttpUrl] = Field(default_factory=dict, description="Router name to /metrics URL.")
    interval_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    timeout_s: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    metrics: list[str] | None = Field(
        default=None,
        description=(
            "Optional exact sample allowlist; by default export all vLLM and router gauges, counters, "
            "and histogram samples, excluding creation timestamps."
        ),
    )

    @model_validator(mode="after")
    def validate_enabled(self) -> "InferenceMetricsConfig":
        if self.enabled and not (self.endpoints or self.router_endpoints):
            raise ValueError("Enabled inference_metrics requires endpoints")
        if self.endpoints.keys() & self.router_endpoints.keys():
            raise ValueError("Replica and router endpoint names must be distinct")
        names = [*self.endpoints, *self.router_endpoints]
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

    def parse(self, replica: str, payload: str, sampled_at: float) -> dict[str, float]:
        """Aggregate labeled series and derive reset-aware counter and histogram rates."""
        result = {}
        gauge_counts = {}
        incomplete_rates = set()
        for family in text_string_to_metric_families(payload):
            if family.type not in {"gauge", "counter", "histogram"}:
                continue
            for sample in family.samples:
                if (
                    not sample.name.startswith(("vllm:", "vllm_router_"))
                    # Prometheus exposes creation timestamps as separate gauges.
                    or sample.name.endswith("_created")
                    or (self.config.metrics is not None and sample.name not in self.config.metrics)
                    or not math.isfinite(sample.value)
                ):
                    continue
                namespace = "router" if sample.name.startswith("vllm_router_") else "vllm"
                name = sample.name.removeprefix("vllm_router_").removeprefix("vllm:")
                labels = urlencode(sorted(sample.labels.items()))
                suffix = "".join(
                    f"/{label}/{value.strip('/')}"
                    for label, value in sorted(sample.labels.items())
                    if label not in {"engine", "model_name"}
                )
                key = f"{namespace}/{replica}/{name}{suffix}"
                series_key = f"{key}/{labels}"
                result[key] = result.get(key, 0.0) + sample.value
                if family.type == "gauge" and name == "kv_cache_usage_perc":
                    gauge_counts[key] = gauge_counts.get(key, 0) + 1
                if family.type in {"counter", "histogram"}:
                    previous = self.previous.get(series_key)
                    self.previous[series_key] = (sampled_at, sample.value)
                    rate_key = f"{namespace}/{replica}/{name.removesuffix('_total')}_per_second{suffix}"
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
        self.add_derived_metrics(result, f"vllm/{replica}/")
        return result

    @staticmethod
    def add_derived_metrics(metrics: dict[str, float], prefix: str) -> None:
        """Derive interval cache hit percentages."""
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
            sampled_at = monotonic()
            metrics = self.parse(replica, payload, sampled_at)
            if not metrics:
                raise ValueError("No matching gauge/counter/histogram samples in metrics response")
            export_metrics(metrics)
            if replica in self.failed:
                logger.info("Inference metrics scraping recovered for replica %s", replica)
                self.failed.remove(replica)
            return metrics
        except Exception as exc:
            if replica not in self.failed:
                # Avoid logging endpoint URLs, which may contain credentials.
                logger.warning(
                    "Inference metrics scrape failed for replica %s (%s); continuing", replica, type(exc).__name__
                )
                self.failed.add(replica)
            return None

    async def run(self, stop: asyncio.Event) -> None:
        """Sample immediately, then periodically until the rollout scope closes."""
        while not stop.is_set():
            snapshots = await asyncio.gather(
                *(self.scrape(name, url) for name, url in self.config.endpoints.items()),
                *(self.scrape(name, url) for name, url in self.config.router_endpoints.items()),
            )
            # Match metric and label paths; missing replicas/series are not zeros.
            replica_metrics = [
                {
                    key.removeprefix(f"vllm/{replica}/"): value
                    for key, value in (snapshot or {}).items()
                    if key.startswith(f"vllm/{replica}/")
                }
                for replica, snapshot in zip(self.config.endpoints, snapshots)
            ]
            shared_metrics = (
                set.intersection(*(set(metrics) for metrics in replica_metrics)) if replica_metrics else set()
            )
            aggregates = {}
            for metric in sorted(shared_metrics):
                values = [metrics[metric] for metrics in replica_metrics]
                total = sum(values)
                name = metric.split("/", 1)[0]
                if name != "prefix_cache_hit_rate":
                    aggregates[f"vllm/total/{metric}"] = total
                aggregates[f"vllm/mean/{metric}"] = total / len(replica_metrics)
                if name in {"kv_cache_usage_perc", "num_requests_waiting"}:
                    aggregates[f"vllm/max/{metric}"] = max(values)
                elif name == "prefix_cache_hit_rate":
                    aggregates[f"vllm/min/{metric}"] = min(values)
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
