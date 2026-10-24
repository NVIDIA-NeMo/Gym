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
import asyncio
import math
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from omegaconf import DictConfig
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import ValidationError

import nemo_gym.inference_metrics as metrics_module
import nemo_gym.server_utils as server_utils
from nemo_gym.exporters.wandb import WandbExporter
from nemo_gym.inference_metrics import InferenceMetricsCollector, InferenceMetricsConfig, collect_inference_metrics


GAUGES = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="test",engine="0"} 3
vllm:num_requests_running{model_name="test",engine="1"} 4
# TYPE unrelated gauge
unrelated 99
"""
MOONCAKE_GAUGES = """# TYPE master_allocated_bytes gauge
master_allocated_bytes 75
# TYPE master_total_capacity_bytes gauge
master_total_capacity_bytes 100
# TYPE segment_allocated_bytes gauge
segment_allocated_bytes{segment="worker0"} 75
# TYPE segment_total_capacity_bytes gauge
segment_total_capacity_bytes{segment="worker0"} 100
"""


def config(**kwargs):
    return InferenceMetricsConfig(enabled=True, endpoints={"replica0": "http://localhost:8000/metrics"}, **kwargs)


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": True},
        {"interval_s": 0},
        {"timeout_s": float("nan")},
        {"endpoints": {"bad/name": "http://localhost/metrics"}},
        {"endpoints": {"replica0": "file:///tmp/metrics"}},
    ],
)
def test_invalid_config(values):
    with pytest.raises(ValidationError):
        InferenceMetricsConfig(**values)


def test_labels_replicas_and_allowlist():
    collector = InferenceMetricsCollector(config())
    values = collector.parse("replica0", GAUGES, 10)
    assert values == {
        "vllm/replica0/num_requests_running": 7,
    }
    assert set(values).isdisjoint(collector.parse("replica1", GAUGES, 10))


def test_mooncake_storage_usage_and_allowlist():
    collector = InferenceMetricsCollector(config(mooncake_endpoint="http://localhost:9003/metrics"))
    assert collector.parse("mooncake", MOONCAKE_GAUGES, 10) == {
        "mooncake/total/allocated_bytes": 75,
        "mooncake/total/total_capacity_bytes": 100,
        "mooncake/total/kv_cache_usage_perc": 0.75,
    }
    assert collector.parse("replica0", MOONCAKE_GAUGES, 10) == {}
    restricted = InferenceMetricsCollector(config(metrics=["master_allocated_bytes"]))
    assert restricted.parse("mooncake", MOONCAKE_GAUGES, 10) == {"mooncake/total/allocated_bytes": 75}
    # Derivation must use this scrape's values, not the previous capacity.
    partial = "# TYPE master_allocated_bytes gauge\nmaster_allocated_bytes 50\n"
    assert collector.parse("mooncake", partial, 12) == {"mooncake/total/allocated_bytes": 50}


@pytest.mark.parametrize("tier", ["", "_mem", "_nof"])
@pytest.mark.parametrize(
    "metric", ["successful_evictions", "attempted_evictions", "evicted_key_count", "evicted_size_bytes"]
)
def test_mooncake_eviction_counters_rates_and_resets(tier, metric):
    # The master mixes native counter names with and without a `_total` suffix.
    native = f"master_{metric}{tier}"
    if not tier and metric in {"successful_evictions", "attempted_evictions"}:
        native += "_total"
    collector = InferenceMetricsCollector(config())

    def payload(value):
        return f"# TYPE {native} counter\n{native} {value}\n"

    total = f"mooncake/total/{metric}{tier}_total"
    rate = f"mooncake/total/{metric}{tier}_per_second"
    assert collector.parse("mooncake", payload(100), 10) == {total: 100}
    assert collector.parse("mooncake", payload(160), 12) == {total: 160, rate: 30}
    assert collector.parse("mooncake", payload(5), 14) == {total: 5}
    assert collector.parse("mooncake", payload(25), 16) == {total: 25, rate: 10}
    assert collector.parse("replica0", payload(25), 16) == {}
    restricted = InferenceMetricsCollector(config(metrics=[native.removesuffix("_total") + "_total"]))
    assert restricted.parse("mooncake", payload(100), 10) == {total: 100}
    assert InferenceMetricsCollector(config(metrics=[])).parse("mooncake", payload(100), 10) == {}


def test_mooncake_eviction_diagnostics_preserve_reasons_and_exclude_segments():
    payload = """# TYPE master_eviction_memory_bytes gauge
master_eviction_memory_bytes 1000
# TYPE master_eviction_blocked_bytes gauge
master_eviction_blocked_bytes{reason="lease"} 600
master_eviction_blocked_bytes{reason="processing"} 200
# TYPE master_eviction_snapshot_timestamp_seconds gauge
master_eviction_snapshot_timestamp_seconds 1234
# TYPE master_offload_pending_bytes gauge
master_offload_pending_bytes 400
# TYPE master_offload_pending_objects gauge
master_offload_pending_objects 2
# TYPE master_offload_oldest_age_seconds gauge
master_offload_oldest_age_seconds 5
# TYPE segment_evicted_size_bytes counter
segment_evicted_size_bytes{segment="worker0"} 999
"""
    collector = InferenceMetricsCollector(config())
    assert collector.parse("mooncake", payload, 10) == {
        "mooncake/total/eviction_memory_bytes": 1000,
        "mooncake/total/eviction_blocked_bytes/reason/lease": 600,
        "mooncake/total/eviction_blocked_bytes/reason/processing": 200,
        "mooncake/total/eviction_snapshot_timestamp_seconds": 1234,
        "mooncake/total/offload_pending_bytes": 400,
        "mooncake/total/offload_pending_objects": 2,
        "mooncake/total/offload_oldest_age_seconds": 5,
    }


@pytest.mark.parametrize("allocated,capacity", [(0, 0), (10, -1), (-1, 100), (10, "NaN"), ("+Inf", 100)])
def test_mooncake_usage_omitted_for_invalid_or_zero_capacity(allocated, capacity):
    collector = InferenceMetricsCollector(config())
    payload = MOONCAKE_GAUGES.replace(" 75", f" {allocated}").replace(" 100", f" {capacity}")
    result = collector.parse("mooncake", payload, 10)
    assert "mooncake/total/kv_cache_usage_perc" not in result
    assert all(math.isfinite(value) for value in result.values())


def test_mooncake_only_config_and_reserved_endpoint_name():
    cfg = InferenceMetricsConfig(enabled=True, mooncake_endpoint="http://localhost:9003/metrics")
    assert cfg.enabled
    for field in ["endpoints", "router_endpoints"]:
        with pytest.raises(ValidationError, match="reserved"):
            InferenceMetricsConfig(
                mooncake_endpoint="http://localhost:9003/metrics", **{field: {"mooncake": "http://localhost/metrics"}}
            )
    with pytest.raises(ValidationError):
        InferenceMetricsConfig(mooncake_endpoint="file:///tmp/metrics")


@pytest.mark.parametrize("only_mooncake", [False, True])
@pytest.mark.parametrize("master_fails", [False, True])
async def test_mooncake_scraped_once_and_does_not_change_vllm_aggregates(monkeypatch, only_mooncake, master_fails):
    cfg = InferenceMetricsConfig(
        enabled=True,
        mooncake_endpoint="http://localhost:9003/metrics",
        endpoints={}
        if only_mooncake
        else {"a": "http://localhost:8001/metrics", "b": "http://localhost:8002/metrics"},
    )
    stop = asyncio.Event()
    published = []

    async def fetch(method, url, **kwargs):
        is_master = ":9003/" in url
        if is_master:
            stop.set()
            if master_fails:
                raise OSError("master unavailable")
        response = MagicMock()
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        response.text = AsyncMock(return_value=MOONCAKE_GAUGES if is_master else GAUGES)
        return response

    fetch_mock = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(metrics_module, "request", fetch_mock)
    monkeypatch.setattr(metrics_module, "export_metrics", published.append)
    await InferenceMetricsCollector(cfg).run(stop)
    assert sum(":9003/" in call.args[1] for call in fetch_mock.call_args_list) == 1
    mooncake_rows = [row for row in published if "mooncake/total/kv_cache_usage_perc" in row]
    assert len(mooncake_rows) == (0 if master_fails else 1)
    if mooncake_rows:
        assert mooncake_rows[0]["mooncake/total/kv_cache_usage_perc"] == 0.75
    if not only_mooncake:
        assert published[-1] == {"vllm/total/num_requests_running": 14, "vllm/mean/num_requests_running": 7}
    assert not any(key.startswith(("vllm/total/allocated", "mooncake/mean/")) for row in published for key in row)


@pytest.mark.parametrize("openmetrics", [False, True])
def test_counter_rates_reset_and_nonfinite(openmetrics):
    collector = InferenceMetricsCollector(config())

    def sample(value, now):
        family_name = "vllm:generation_tokens" if openmetrics else "vllm:generation_tokens_total"
        payload = f"# TYPE {family_name} counter\nvllm:generation_tokens_total {value}\n"
        if openmetrics:
            payload += "# EOF\n"
        return collector.parse(
            "replica0",
            payload,
            now,
            content_type="application/openmetrics-text; version=1.0.0; charset=utf-8" if openmetrics else "text/plain",
        )

    rate = "vllm/replica0/generation_tokens_per_second"
    assert rate not in sample(100, 10)
    assert sample(160, 12)[rate] == 30
    assert rate not in sample(5, 14)
    assert sample(25, 16)[rate] == 10
    if openmetrics:
        # OpenMetrics rejects NaN counters at the parser level; scrape handles
        # malformed responses through its existing failure/recovery path.
        with pytest.raises(ValueError, match="Counter-like samples cannot be NaN"):
            sample("NaN", 17)
    else:
        assert sample("NaN", 17) == {}


async def test_mixed_frontend_http_counters_publish_total_generation_rate(monkeypatch):
    """Rust OpenMetrics and Python Prometheus counters must share Gym metric names."""
    app = web.Application()
    scrapes = {"rust": 0, "python": 0}

    async def serve(request):
        replica = request.match_info["replica"]
        index = scrapes[replica]
        scrapes[replica] += 1
        value = (100, 160)[index] if replica == "rust" else (200, 300)[index]
        family = "vllm:generation_tokens" if replica == "rust" else "vllm:generation_tokens_total"
        payload = f'# TYPE {family} counter\nvllm:generation_tokens_total{{model_name="test",engine="0"}} {value}\n'
        if replica == "rust":
            return web.Response(
                text=payload + "# EOF\n",
                headers={"Content-Type": "application/openmetrics-text; version=1.0.0; charset=utf-8"},
            )
        return web.Response(text=payload, content_type="text/plain")

    app.router.add_get("/{replica}/metrics", serve)
    stop = asyncio.Event()
    published = {}

    def publish(metrics):
        published.update(metrics)
        if "vllm/total/generation_tokens_per_second" in metrics:
            stop.set()

    sampled_times = iter([10, 10, 12, 12])
    monkeypatch.setattr(metrics_module, "monotonic", lambda: next(sampled_times))
    monkeypatch.setattr(metrics_module, "export_metrics", publish)
    async with TestServer(app) as server, ClientSession() as session:
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)
        cfg = InferenceMetricsConfig(
            enabled=True,
            endpoints={replica: str(server.make_url(f"/{replica}/metrics")) for replica in scrapes},
            interval_s=0.001,
        )
        collector = InferenceMetricsCollector(cfg)
        await asyncio.wait_for(collector.run(stop), timeout=2)

    assert not collector.failed
    assert scrapes == {"rust": 2, "python": 2}
    assert published["vllm/rust/generation_tokens_total"] == 160
    assert published["vllm/rust/generation_tokens_per_second"] == 30
    assert published["vllm/python/generation_tokens_per_second"] == 50
    assert published["vllm/total/generation_tokens_per_second"] == 80
    assert published["vllm/mean/generation_tokens_per_second"] == 40


@pytest.mark.parametrize(
    "metric_prefix,namespace", [("vllm:", "vllm"), ("sglang:", "sglang"), ("vllm_router_", "router")]
)
def test_histogram_samples_rates_resets_and_allowlist(metric_prefix, namespace):
    collector = InferenceMetricsCollector(config())
    metric_name = metric_prefix + "iteration_tokens_total"
    prefix = f"{namespace}/replica0/iteration_tokens_total"

    def payload(observations):
        registry = CollectorRegistry()
        histogram = Histogram(metric_name, "Tokens per step", ["engine"], buckets=[32], registry=registry)
        for value in observations:
            histogram.labels(engine="0").observe(value)
        histogram.labels(engine="1").observe(64)
        Counter(metric_prefix + "requests_total", "Requests", registry=registry).inc()
        return generate_latest(registry).decode()

    first_payload = payload([8, 32])
    first = collector.parse("replica0", first_payload, 10)
    assert first == {
        prefix + "_sum": 104,
        prefix + "_count": 3,
        prefix + "_bucket/le/32.0": 2,
        prefix + "_bucket/le/+Inf": 3,
        f"{namespace}/replica0/requests_total": 1,
    }
    second = collector.parse("replica0", payload([8, 32, 16, 64]), 12)
    assert second[prefix + "_sum"] == 184
    assert second[prefix + "_count"] == 5
    assert second[prefix + "_sum_per_second"] == 40
    assert second[prefix + "_count_per_second"] == 1
    assert second[prefix + "_bucket_per_second/le/32.0"] == 0.5
    assert second[prefix + "_bucket_per_second/le/+Inf"] == 1
    reset = collector.parse("replica0", payload([1]), 14)
    assert reset[prefix + "_sum"] == 65
    assert not any("iteration_tokens_total" in key and "per_second" in key for key in reset)
    recovered = collector.parse("replica0", payload([1, 8]), 16)
    assert recovered[prefix + "_sum_per_second"] == 4
    assert recovered[prefix + "_count_per_second"] == 0.5

    restricted = InferenceMetricsCollector(config(metrics=[metric_name + "_sum", metric_name + "_created"]))
    assert restricted.parse("replica0", first_payload, 10) == {prefix + "_sum": 104}


@pytest.mark.parametrize("namespace", ["vllm", "sglang"])
async def test_real_http_scrape_publishes(monkeypatch, namespace):
    app = web.Application()

    async def serve(request):
        return web.Response(text=GAUGES.replace("vllm:", namespace + ":"))

    app.router.add_get("/metrics", serve)
    publish = MagicMock()
    monkeypatch.setattr(metrics_module, "export_metrics", publish)
    async with TestServer(app) as server, ClientSession() as session:
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)
        cfg = InferenceMetricsConfig(enabled=True, endpoints={"replica0": str(server.make_url("/metrics"))})
        await InferenceMetricsCollector(cfg).scrape("replica0", cfg.endpoints["replica0"])
    payload = publish.call_args.args[0]
    assert payload[f"{namespace}/replica0/num_requests_running"] == 7
    assert all(key.startswith(f"{namespace}/replica0/") for key in payload)
    assert publish.call_args.kwargs == {}


async def test_timeout_is_bounded_and_warns_once(monkeypatch, caplog):
    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(metrics_module, "request", stalled)
    collector = InferenceMetricsCollector(config(timeout_s=0.01))
    for _ in range(2):
        await asyncio.wait_for(collector.scrape("replica0", collector.config.endpoints["replica0"]), timeout=1)
    assert caplog.text.count("Inference metrics scrape failed") == 1


@pytest.mark.parametrize("mode", ["success", "failure", "cancel"])
async def test_sampler_lifecycle(monkeypatch, mode):
    started, finished = asyncio.Event(), asyncio.Event()

    async def run(self, stop):
        started.set()
        try:
            await stop.wait()
        finally:
            finished.set()

    monkeypatch.setattr(metrics_module, "get_exporters", lambda: [object()])
    monkeypatch.setattr(InferenceMetricsCollector, "run", run)

    async def body():
        async with collect_inference_metrics(config()):
            await started.wait()
            if mode == "failure":
                raise RuntimeError("rollout failed")
            if mode == "cancel":
                await asyncio.Event().wait()

    task = asyncio.create_task(body())
    await started.wait()
    if mode == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif mode == "failure":
        with pytest.raises(RuntimeError, match="rollout failed"):
            await task
    else:
        await task
    assert finished.is_set()
    assert not any(t.get_name() == "inference-metrics" for t in asyncio.all_tasks())


async def test_disabled_or_no_exporter_does_not_scrape(monkeypatch):
    scrape = AsyncMock()
    monkeypatch.setattr(InferenceMetricsCollector, "scrape", scrape)
    monkeypatch.setattr(metrics_module, "get_exporters", lambda: [])
    for cfg in [InferenceMetricsConfig(), config()]:
        async with collect_inference_metrics(cfg):
            await asyncio.sleep(0)
    scrape.assert_not_called()


def test_wandb_progress_axis_and_automatic_steps(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr("nemo_gym.exporters.wandb.wandb.init", MagicMock(return_value=run))
    exporter = WandbExporter(
        DictConfig(
            {
                "wandb_project": "test",
                "wandb_name": "test",
                "wandb_api_key": "test",
                "results_dir": "/tmp",
                "inference_metrics": {"enabled": True},
            }
        )
    )
    exporter.setup()
    assert run.define_metric.call_count == 2
    run.define_metric.assert_any_call("progress/*", step_metric="progress/completion_pct")
    for payload in [
        {"progress/completion_pct": 1},
        {"vllm/replica0/num_requests_running": 2},
        {"progress/completion_pct": 2},
    ]:
        exporter.export_metrics(payload)
    assert all(call.kwargs == {"step": None, "commit": True} for call in run.log.call_args_list)


async def test_failed_replica_does_not_block_healthy_replica(monkeypatch, caplog):
    cfg = InferenceMetricsConfig(
        enabled=True,
        endpoints={
            "healthy": "http://localhost:8000/metrics",
            "broken": "http://localhost:8001/metrics",
        },
        interval_s=0.01,
    )
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    response.text = AsyncMock(return_value=GAUGES)

    async def fetch(method, url, **kwargs):
        if ":8001/" in url:
            raise OSError("unavailable")
        return response

    stop = asyncio.Event()
    published = []

    def publish(metrics):
        published.append(metrics)
        if len(published) == 2:
            stop.set()

    monkeypatch.setattr(metrics_module, "request", fetch)
    monkeypatch.setattr(metrics_module, "export_metrics", publish)
    await asyncio.wait_for(InferenceMetricsCollector(cfg).run(stop), timeout=1)
    assert len(published) == 2
    assert all("vllm/healthy/num_requests_running" in row for row in published)
    assert caplog.text.count("Inference metrics scrape failed") == 1


async def test_empty_scrape_warns_then_recovers(monkeypatch, caplog):
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)
    response.text = AsyncMock(
        side_effect=[
            '# TYPE latency histogram\nlatency_bucket{le="+Inf"} 1\nlatency_sum 1\nlatency_count 1\n',
            GAUGES,
        ]
    )
    monkeypatch.setattr(metrics_module, "request", AsyncMock(return_value=response))
    publish = MagicMock()
    monkeypatch.setattr(metrics_module, "export_metrics", publish)
    collector = InferenceMetricsCollector(config())
    await collector.scrape("replica0", collector.config.endpoints["replica0"])
    assert "replica0" in collector.failed
    publish.assert_not_called()
    await collector.scrape("replica0", collector.config.endpoints["replica0"])
    assert not collector.failed
    publish.assert_called_once()
    assert "Inference metrics scrape failed" in caplog.text


def test_all_metrics_and_remaining_label_paths():
    collector = InferenceMetricsCollector(config())
    payload = """# TYPE vllm:spec_tokens_total counter
vllm:spec_tokens_total{engine="0",model_name="test",position="2"} 10
vllm:spec_tokens_total{engine="1",model_name="test",position="2"} 20
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0"} 0.2
vllm:kv_cache_usage_perc{engine="1"} 0.4
# TYPE vllm:latency histogram
vllm:latency_bucket{le="1"} 4
vllm:latency_sum 4
vllm:latency_count 4
"""
    first = collector.parse("replica0", payload, 10)
    assert first == {
        "vllm/replica0/spec_tokens_total/position/2": 30,
        "vllm/replica0/kv_cache_usage_perc": pytest.approx(0.3),
        "vllm/replica0/latency_bucket/le/1": 4,
        "vllm/replica0/latency_sum": 4,
        "vllm/replica0/latency_count": 4,
    }
    second = collector.parse("replica0", payload.replace("} 10", "} 14").replace("} 20", "} 26"), 12)
    assert second["vllm/replica0/spec_tokens_per_second/position/2"] == 5
    reset = collector.parse("replica0", payload.replace("} 10", "} 1"), 14)
    assert "vllm/replica0/spec_tokens_per_second/position/2" not in reset
    restricted = InferenceMetricsCollector(config(metrics=["vllm:num_requests_running"]))
    assert restricted.parse("replica0", payload, 10) == {}


@pytest.mark.parametrize("missing", [False, True])
async def test_aggregates_require_all_fresh_replica_samples(monkeypatch, missing):
    cfg = InferenceMetricsConfig(
        enabled=True,
        endpoints={
            "a": "http://localhost:8000/metrics",
            "b": "http://localhost:8001/metrics",
        },
    )
    collector = InferenceMetricsCollector(cfg)
    stop = asyncio.Event()

    async def scrape(replica, url):
        stop.set()
        if replica == "b" and missing:
            return None
        return {
            f"vllm/{replica}/generation_tokens_per_second": 100 if replica == "a" else 200,
            f"vllm/{replica}/num_requests_running": 0 if replica == "a" else 4,
            f"vllm/{replica}/num_requests_waiting": 8 if replica == "a" else 2,
            f"vllm/{replica}/spec_tokens_total/position/2": 10 if replica == "a" else 30,
            f"vllm/{replica}/kv_cache_usage_perc": 0.2 if replica == "a" else 0.6,
            f"vllm/{replica}/only_on_{replica}": 99,
        }

    monkeypatch.setattr(collector, "scrape", scrape)
    publish = MagicMock()
    monkeypatch.setattr(metrics_module, "export_metrics", publish)
    await collector.run(stop)
    if missing:
        publish.assert_not_called()
    else:
        assert publish.call_args.args[0]["vllm/total/generation_tokens_per_second"] == 300
        assert publish.call_args.args[0]["vllm/mean/generation_tokens_per_second"] == 150

        payload = publish.call_args.args[0]
        assert payload["vllm/total/num_requests_running"] == 4
        assert payload["vllm/mean/num_requests_running"] == 2
        assert payload["vllm/total/spec_tokens_total/position/2"] == 40
        assert payload["vllm/mean/spec_tokens_total/position/2"] == 20
        assert payload["vllm/mean/kv_cache_usage_perc"] == pytest.approx(0.4)
        assert payload["vllm/max/kv_cache_usage_perc"] == 0.6
        assert payload["vllm/total/num_requests_waiting"] == 10
        assert payload["vllm/mean/num_requests_waiting"] == 5
        assert payload["vllm/max/num_requests_waiting"] == 8
        assert not any("only_on_" in key for key in payload)
        assert len(payload) == 12


async def test_histogram_only_scrapes_publish_replica_and_aggregate_samples(monkeypatch):
    cfg = InferenceMetricsConfig(
        enabled=True,
        endpoints={"a": "http://localhost:8000/metrics", "b": "http://localhost:8001/metrics"},
    )
    stop = asyncio.Event()
    published = {}

    async def fetch(method, url, **kwargs):
        registry = CollectorRegistry()
        histogram = Histogram("vllm:iteration_tokens_total", "Tokens per step", buckets=[32], registry=registry)
        for value in [8, 32] if ":8000/" in url else [64]:
            histogram.observe(value)
        response = MagicMock()
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        response.text = AsyncMock(return_value=generate_latest(registry).decode())
        return response

    def publish(metrics):
        published.update(metrics)
        stop.set()

    monkeypatch.setattr(metrics_module, "request", fetch)
    monkeypatch.setattr(metrics_module, "export_metrics", publish)
    collector = InferenceMetricsCollector(cfg)
    await collector.run(stop)
    assert not collector.failed
    for replica, expected in {"a": (40, 2, 2), "b": (64, 1, 0), "total": (104, 3, 2), "mean": (52, 1.5, 1)}.items():
        total, count, bucket = expected
        prefix = f"vllm/{replica}/iteration_tokens_total"
        assert published[prefix + "_sum"] == total
        assert published[prefix + "_count"] == count
        assert published[prefix + "_bucket/le/32.0"] == bucket
        assert published[prefix + "_bucket/le/+Inf"] == count
    assert not any("_created" in key for key in published)


def test_dbo_metrics_preserve_engine_outcome_and_microbatch_labels():
    collector = InferenceMetricsCollector(config())
    registry = CollectorRegistry()
    steps = Counter("vllm:dbo_steps_total", "Steps", ["model_name", "engine", "outcome"], registry=registry)
    sizes = Histogram(
        "vllm:dbo_microbatch_tokens", "Tokens", ["model_name", "engine", "microbatch"], registry=registry
    )
    steps.labels("model", "0", "executed").inc(2)
    steps.labels("model", "1", "below_threshold").inc(3)
    sizes.labels("model", "0", "0").observe(100)
    sizes.labels("model", "1", "0").observe(200)
    first = collector.parse("prefill0", generate_latest(registry).decode(), 10)
    prefix = "vllm/prefill0/dbo_"
    assert first[prefix + "steps_total/engine/0/outcome/executed"] == 2
    assert first[prefix + "steps_total/engine/1/outcome/below_threshold"] == 3
    assert first[prefix + "microbatch_tokens_sum/engine/0/microbatch/0"] == 100
    assert first[prefix + "microbatch_tokens_sum/engine/1/microbatch/0"] == 200
    assert first[prefix + "microbatch_tokens_bucket/engine/0/le/+Inf/microbatch/0"] == 1
    steps.labels("model", "0", "executed").inc(4)
    second = collector.parse("prefill0", generate_latest(registry).decode(), 12)
    assert second[prefix + "steps_per_second/engine/0/outcome/executed"] == 2
    assert second[prefix + "steps_per_second/engine/1/outcome/below_threshold"] == 0


def test_derived_cache_and_source_prompt_metrics():
    collector = InferenceMetricsCollector(config())

    def scrape(hits, queries, computed, now):
        payload = f"""# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total {hits}
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total {queries}
# TYPE vllm:prompt_tokens_by_source_total counter
vllm:prompt_tokens_by_source_total{{source="local_compute"}} {computed}
vllm:prompt_tokens_by_source_total{{source="local_cache_hit"}} 9999
"""
        return collector.parse("replica0", payload, now)

    cache = "vllm/replica0/prefix_cache_hit_rate"
    new = "vllm/replica0/prompt_tokens_by_source_per_second/source/local_compute"
    first = scrape(100, 200, 40, 10)
    assert cache not in first and new not in first
    second = scrape(180, 300, 70, 12)
    assert second[cache] == 80
    assert second[new] == 15
    idle = scrape(180, 300, 70, 14)
    assert cache not in idle and idle[new] == 0
    reset = scrape(1, 2, 1, 16)
    assert cache not in reset and new not in reset


async def test_total_cache_hit_rate_is_query_weighted(monkeypatch):
    cfg = InferenceMetricsConfig(
        enabled=True,
        endpoints={
            "a": "http://localhost:8000/metrics",
            "b": "http://localhost:8001/metrics",
        },
    )
    collector = InferenceMetricsCollector(cfg)
    stop = asyncio.Event()

    async def scrape(replica, url):
        stop.set()
        hits, queries = (90, 100) if replica == "a" else (10, 20)
        result = {
            f"vllm/{replica}/prefix_cache_hits_per_second": hits,
            f"vllm/{replica}/prefix_cache_queries_per_second": queries,
            f"vllm/{replica}/prompt_tokens_by_source_per_second/source/local_compute": 5,
        }
        collector.add_derived_metrics(result, f"vllm/{replica}/")
        return result

    monkeypatch.setattr(collector, "scrape", scrape)
    publish = MagicMock()
    monkeypatch.setattr(metrics_module, "export_metrics", publish)
    await collector.run(stop)
    result = publish.call_args.args[0]
    assert result["vllm/total/prefix_cache_hit_rate"] == pytest.approx(100 * 100 / 120)
    assert result["vllm/mean/prefix_cache_hit_rate"] == 70
    assert result["vllm/min/prefix_cache_hit_rate"] == 50
    assert result["vllm/total/prompt_tokens_by_source_per_second/source/local_compute"] == 10
    assert result["vllm/mean/prompt_tokens_by_source_per_second/source/local_compute"] == 5


def test_router_counters_labels_and_reset():
    collector = InferenceMetricsCollector(config())

    def sample(value, now):
        return collector.parse(
            "main",
            "# TYPE vllm_router_processed_requests_total counter\n"
            f'vllm_router_processed_requests_total{{worker="http://worker:8001"}} {value}\n'
            "# TYPE vllm_router_worker_load gauge\n"
            'vllm_router_worker_load{worker="http://worker:8001"} 12\n',
            now,
        )

    prefix = "router/main/"
    suffix = "/worker/http://worker:8001"
    rate = prefix + "processed_requests_per_second" + suffix
    first = sample(100, 10)
    assert first[prefix + "worker_load" + suffix] == 12
    assert rate not in first
    assert sample(160, 12)[rate] == 30
    assert rate not in sample(5, 14)
    assert sample(25, 16)[rate] == 10


async def test_router_endpoint_does_not_remove_replica_aggregates(monkeypatch):
    cfg = config(router_endpoints={"main": "http://localhost:29000/metrics"})
    collector = InferenceMetricsCollector(cfg)
    stop = asyncio.Event()
    publish = MagicMock()
    monkeypatch.setattr(metrics_module, "export_metrics", publish)

    async def scrape(name, url):
        if name == "main":
            stop.set()
            return {"router/main/worker_load/worker/backend": 20}
        return {f"vllm/{name}/num_requests_running": 7}

    collector.scrape = AsyncMock(side_effect=scrape)
    await collector.run(stop)
    assert collector.scrape.await_count == 2
    publish.assert_called_once_with({"vllm/total/num_requests_running": 7, "vllm/mean/num_requests_running": 7})


def test_router_only_and_endpoint_validation():
    cfg = InferenceMetricsConfig(enabled=True, router_endpoints={"main": "http://localhost:29000/metrics"})
    assert cfg.enabled
    with pytest.raises(ValidationError):
        config(router_endpoints={"replica0": "http://localhost:29000/metrics"})
    with pytest.raises(ValidationError):
        config(router_endpoints={"bad/name": "http://localhost:29000/metrics"})


def test_router_routes_are_readable_metric_paths():
    collector = InferenceMetricsCollector(config())
    payload = (
        '# TYPE vllm_router_requests_total counter\nvllm_router_requests_total{route="/v1/chat/completions"} 10\n'
    )
    first = collector.parse("main", payload, 10)
    assert first == {"router/main/requests_total/route/v1/chat/completions": 10}
    second = collector.parse("main", payload.replace(" 10", " 20"), 12)
    assert second["router/main/requests_per_second/route/v1/chat/completions"] == 5


def test_sglang_rates_and_rank_labels():
    collector = InferenceMetricsCollector(config())
    payload = """# TYPE sglang:realtime_tokens_total counter
sglang:realtime_tokens_total{model_name="test",engine_type="decode",dp_rank="0",mode="decode"} 100
sglang:realtime_tokens_total{model_name="test",engine_type="decode",dp_rank="1",mode="decode"} 200
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="test",engine_type="decode",dp_rank="0"} 0.2
sglang:token_usage{model_name="test",engine_type="decode",dp_rank="1"} 0.8
"""
    first = collector.parse("decode0", payload, 10)
    assert first["sglang/decode0/token_usage/dp_rank/0"] == 0.2
    assert first["sglang/decode0/token_usage/dp_rank/1"] == 0.8
    assert not any("per_second" in key for key in first)
    second = collector.parse("decode0", payload.replace(" 100", " 120").replace(" 200", " 260"), 12)
    assert second["sglang/decode0/realtime_tokens_per_second/dp_rank/0/mode/decode"] == 10
    assert second["sglang/decode0/realtime_tokens_per_second/dp_rank/1/mode/decode"] == 30
    reset = collector.parse("decode0", payload, 14)
    assert not any("per_second" in key for key in reset)


@pytest.mark.parametrize("missing", [False, True])
async def test_sglang_aggregates_counts_but_not_ratios(monkeypatch, missing):
    cfg = InferenceMetricsConfig(
        enabled=True,
        endpoints={"prefill0": "http://localhost:8000/metrics", "decode0": "http://localhost:8001/metrics"},
    )
    collector = InferenceMetricsCollector(cfg)
    stop = asyncio.Event()

    async def scrape(replica, url):
        stop.set()
        if missing and replica == "decode0":
            return None
        role, count, usage = ("prefill", 3, 0.2) if replica == "prefill0" else ("decode", 7, 0.6)
        payload = f'''# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{{model_name="test",engine_type="{role}",tp_rank="0"}} {count}
# TYPE sglang:token_usage gauge
sglang:token_usage{{model_name="test",engine_type="{role}",tp_rank="0"}} {usage}
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{{model_name="test",engine_type="{role}",tp_rank="0"}} {usage}
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{{model_name="test",engine_type="{role}",tp_rank="0"}} {count}
# TYPE sglang:generation_tokens_total counter
sglang:generation_tokens_total{{model_name="test",engine_type="{role}",is_streaming="True"}} 100
'''
        collector.parse(replica, payload, 10)
        return collector.parse(replica, payload.replace(" 100", f" {100 + count * 2}"), 12)

    monkeypatch.setattr(collector, "scrape", scrape)
    publish = MagicMock()
    monkeypatch.setattr(metrics_module, "export_metrics", publish)
    await collector.run(stop)
    if missing:
        publish.assert_not_called()
        return
    result = publish.call_args.args[0]
    assert result["sglang/total/num_running_reqs/tp_rank/0"] == 10
    assert result["sglang/total/generation_tokens_per_second/is_streaming/True"] == 10
    assert result["sglang/max/num_queue_reqs/tp_rank/0"] == 7
    for name in ("token_usage", "cache_hit_rate"):
        assert f"sglang/total/{name}/tp_rank/0" not in result
        assert result[f"sglang/mean/{name}/tp_rank/0"] == pytest.approx(0.4)
        assert result[f"sglang/max/{name}/tp_rank/0"] == 0.6
    assert result["sglang/min/cache_hit_rate/tp_rank/0"] == 0.2
    assert all(key.startswith("sglang/") for key in result)


INVENTORY = """# TYPE vllm:mooncake_cache_layout gauge
vllm:mooncake_cache_layout{scope="model",prefix="model@rank0",kind="mamba"} 1
vllm:mooncake_cache_layout{scope="model",prefix="model@rank1",kind="mamba"} 1
vllm:mooncake_cache_layout{scope="model",prefix="lineage",kind="mamba_lineage"} 1
vllm:mooncake_cache_layout{scope="model",prefix="retired",kind="mamba_retired"} 1
# TYPE vllm:cache_inventory gauge
vllm:cache_inventory{engine="0",kind="mamba",measure="distinct_prefixes"} 12
vllm:cache_inventory{engine="1",kind="mamba",measure="distinct_prefixes"} 8
"""


async def test_inventory_default_off_never_requests_master(monkeypatch):
    collector = InferenceMetricsCollector(config(mooncake_endpoint="http://localhost:9003/metrics"))
    request = AsyncMock()
    monkeypatch.setattr(metrics_module, "request", request)
    assert collector.parse("replica0", GAUGES + INVENTORY, 10) == {"vllm/replica0/num_requests_running": 7}
    assert not collector.cache_layouts
    await collector.report_cache_inventory()
    request.assert_not_called()


async def test_inventory_http_expected_layout_and_zero_counts(monkeypatch):
    app = web.Application()
    bodies = []

    async def serve(request):
        bodies.append(await request.json())
        return web.json_response(
            {
                "counts": {
                    "mamba.complete_prefixes": 0,
                    "mamba.incomplete_prefixes": 3,
                    "mamba.retired.incomplete_prefixes": 2,
                    "mamba.retired.ready_bytes": 4096,
                    "mamba.anchor_or_unknown.incomplete_prefixes": 1,
                    "mamba.tip.complete_prefixes": 0,
                },
                "snapshot_timestamp_seconds": 123,
                "scan_seconds": 0.01,
            }
        )

    app.router.add_post("/cache_inventory", serve)
    publish = MagicMock()
    monkeypatch.setattr(metrics_module, "export_metrics", publish)
    async with TestServer(app) as server, ClientSession() as session:
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)
        collector = InferenceMetricsCollector(
            config(cache_inventory=True, mooncake_endpoint=str(server.make_url("/metrics")))
        )
        values = collector.parse("replica0", INVENTORY, 10)
        assert values == {
            "vllm/replica0/cache_inventory/engine/0/kind/mamba/measure/distinct_prefixes": 12,
            "vllm/replica0/cache_inventory/engine/1/kind/mamba/measure/distinct_prefixes": 8,
        }
        collector.parse("replica1", INVENTORY, 10)
        await collector.report_cache_inventory()
    assert bodies == [
        {
            "tenant_id": "default",
            "namespaces": [
                {"scope": "model", "prefix": "lineage", "kind": "mamba_lineage"},
                {"scope": "model", "prefix": "model@rank0", "kind": "mamba"},
                {"scope": "model", "prefix": "model@rank1", "kind": "mamba"},
                {"scope": "model", "prefix": "retired", "kind": "mamba_retired"},
            ],
        }
    ]
    assert publish.call_args.args[0] == {
        "mooncake/total/cache_inventory/mamba/complete_prefixes": 0,
        "mooncake/total/cache_inventory/mamba/incomplete_prefixes": 3,
        "mooncake/total/cache_inventory/mamba/retired/incomplete_prefixes": 2,
        "mooncake/total/cache_inventory/mamba/retired/ready_bytes": 4096,
        "mooncake/total/cache_inventory/mamba/anchor_or_unknown/incomplete_prefixes": 1,
        "mooncake/total/cache_inventory/mamba/tip/complete_prefixes": 0,
        "mooncake/total/cache_inventory/snapshot_timestamp_seconds": 123,
        "mooncake/total/cache_inventory/scan_seconds": 0.01,
    }


async def test_slow_inventory_does_not_block_scrapes_and_is_cancelled(monkeypatch):
    collector = InferenceMetricsCollector(
        config(cache_inventory=True, interval_s=0.001, mooncake_endpoint="http://localhost:9003/metrics")
    )
    stop, started, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    samples = 0

    async def scrape(replica, url):
        nonlocal samples
        if replica == "replica0":
            samples += 1
            collector.parse(replica, INVENTORY, samples)
            if samples == 3:
                stop.set()
        return {"vllm/replica0/num_requests_running": 1}

    async def inventory():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(collector, "scrape", scrape)
    monkeypatch.setattr(collector, "report_cache_inventory", inventory)
    monkeypatch.setattr(metrics_module, "export_metrics", MagicMock())
    await asyncio.wait_for(collector.run(stop), 1)
    assert samples == 3
    assert started.is_set() and cancelled.is_set()
    assert collector._inventory_task is None


def prefill_counter_payload(values: dict[str, int], *, reason: str = "mamba_alignment") -> str:
    registry = CollectorRegistry()
    for name, value in values.items():
        Counter(f"vllm:prefill_{name}_total", "Batching interval inputs", registry=registry).inc(value)
    reasons = Counter(
        "vllm:prefill_underfilled_batches_by_reason_total", "Overlapping constraints", ["reason"], registry=registry
    )
    reasons.labels(reason=reason).inc(values.get("underfilled_batches", 0))
    return generate_latest(registry).decode()


@pytest.mark.parametrize(
    "reason", ["mamba_alignment", "mamba_alignment_budget_too_small", "mamba_alignment_shared_prefix"]
)
def test_prefill_interval_averages_use_deltas_and_omit_idle_or_reset_intervals(reason: str):
    collector = InferenceMetricsCollector(config())
    values = dict(
        completed_requests=2,
        request_forwards=6,
        request_tokens=620,
        request_forward_observations=6,
        batch_tokens=640,
        batch_requests=7,
        batches=4,
        unused_token_slots=384,
        underfilled_batches=2,
    )
    prefix = "vllm/replica0/"
    first = collector.parse("replica0", prefill_counter_payload(dict.fromkeys(values, 0), reason=reason), 10)
    assert prefix + "prefill_forwards_per_request" not in first
    result = collector.parse("replica0", prefill_counter_payload(values, reason=reason), 20)
    assert result[prefix + "prefill_forwards_per_request"] == pytest.approx(3)
    assert result[prefix + "prefill_tokens_per_request_forward"] == pytest.approx(620 / 6)
    assert result[prefix + "prefill_tokens_per_forward"] == 160
    assert result[prefix + "prefill_requests_per_forward"] == pytest.approx(1.75)
    assert result[prefix + "prefill_batch_token_fill_ratio"] == 0.625
    assert result[prefix + "prefill_underfilled_batch_fraction"] == 0.5
    assert result[prefix + f"prefill_underfill_reason_fraction/reason/{reason}"] == 0.5
    assert not any("bucket" in key for key in result)
    idle = collector.parse("replica0", prefill_counter_payload(values, reason=reason), 30)
    assert prefix + "prefill_tokens_per_forward" not in idle
    reset = collector.parse("replica0", prefill_counter_payload(dict.fromkeys(values, 0), reason=reason), 40)
    assert prefix + "prefill_tokens_per_forward" not in reset


async def test_prefill_global_average_pools_counts_instead_of_averaging_engine_means(monkeypatch):
    cfg = InferenceMetricsConfig(
        enabled=True,
        endpoints={
            "a": "http://localhost:8000/metrics",
            "b": "http://localhost:8001/metrics",
        },
    )
    collector = InferenceMetricsCollector(cfg)
    stop = asyncio.Event()
    published = {}

    async def scrape(replica, url):
        tokens, batches = (100, 1) if replica == "a" else (1800, 9)
        result = {
            f"vllm/{replica}/prefill_batch_tokens_per_second": tokens,
            f"vllm/{replica}/prefill_batches_per_second": batches,
            f"vllm/{replica}/prefill_tokens_per_forward": tokens / batches,
            f"vllm/{replica}/prefill_kv_lookup_duration_ns_per_second": tokens * 1e6,
            f"vllm/{replica}/prefill_kv_lookup_completed_per_second": batches,
            f"vllm/{replica}/prefill_kv_lookup_latency_ms": tokens / batches,
            f"vllm/{replica}/prefill_delay_steps_per_second": batches,
            f"vllm/{replica}/prefill_blocked_requests_by_reason_per_second/reason/kv_lookup_pending": tokens,
            f"vllm/{replica}/prefill_blocked_requests_per_step/reason/kv_lookup_pending": tokens / batches,
            f"vllm/{replica}/prefill_readiness_steps_per_second/engine/0": batches,
            f"vllm/{replica}/prefill_ready_requests_per_second/engine/0": tokens,
            f"vllm/{replica}/prefill_ready_requests_per_step/engine/0": tokens / batches,
            f"vllm/{replica}/prefill_ready_requests_per_step": tokens / batches,
        }
        result[f"vllm/{replica}/prefill_kv_lookup_timed_completed_per_second"] = batches
        for phase, multiplier in (("queue", 1), ("execution", 2), ("observation", 3)):
            result[f"vllm/{replica}/prefill_kv_lookup_{phase}_duration_ns_per_second"] = tokens * multiplier * 1e6
            result[f"vllm/{replica}/prefill_kv_lookup_{phase}_latency_ms"] = tokens * multiplier / batches
        result[f"vllm/{replica}/prefill_kv_lookup_client_latency_ms"] = tokens * 6 / batches
        for name, (numerator, denominator, label) in metrics_module.PREFILL_OUTCOME_METRICS.items():
            result.update(
                {
                    f"vllm/{replica}/prefill_{denominator}_per_second/engine/0": batches,
                    f"vllm/{replica}/prefill_{numerator}_per_second/engine/0/{label}/mamba_alignment": tokens,
                    f"vllm/{replica}/{name}/engine/0/{label}/mamba_alignment": tokens / batches,
                    f"vllm/{replica}/{name}/{label}/mamba_alignment": tokens / batches,
                }
            )
        return result

    def publish(metrics):
        published.update(metrics)
        stop.set()

    monkeypatch.setattr(collector, "scrape", scrape)
    monkeypatch.setattr(metrics_module, "export_metrics", publish)
    await collector.run(stop)
    assert published["vllm/total/prefill_tokens_per_forward"] == 190
    assert "vllm/mean/prefill_tokens_per_forward" not in published
    assert published["vllm/total/prefill_kv_lookup_latency_ms"] == 190
    for phase, multiplier in (("queue", 1), ("execution", 2), ("observation", 3), ("client", 6)):
        assert published[f"vllm/total/prefill_kv_lookup_{phase}_latency_ms"] == 190 * multiplier
        assert f"vllm/mean/prefill_kv_lookup_{phase}_latency_ms" not in published
    assert published["vllm/total/prefill_blocked_requests_per_step/reason/kv_lookup_pending"] == 190
    assert "vllm/mean/prefill_kv_lookup_latency_ms" not in published
    assert "vllm/mean/prefill_blocked_requests_per_step/reason/kv_lookup_pending" not in published
    assert published["vllm/total/prefill_ready_requests_per_step"] == 190
    assert published["vllm/total/prefill_ready_requests_per_step/engine/0"] == 190
    assert "vllm/mean/prefill_ready_requests_per_step" not in published
    for name, (_, _, label) in metrics_module.PREFILL_OUTCOME_METRICS.items():
        assert published[f"vllm/total/{name}/{label}/mamba_alignment"] == 190
        assert published[f"vllm/total/{name}/engine/0/{label}/mamba_alignment"] == 190
        assert f"vllm/mean/{name}/{label}/mamba_alignment" not in published


def test_prefill_lookup_latency_and_request_counts_use_interval_deltas():
    collector = InferenceMetricsCollector(config())
    registry = CollectorRegistry()
    counters = {
        name: Counter(f"vllm:prefill_{name}_total", "Lookup inputs", registry=registry)
        for name in (
            "kv_lookup_completed",
            "kv_lookup_duration_ns",
            "kv_lookup_pending_completed",
            "kv_lookup_pending_duration_ns",
            "delay_steps",
        )
    }
    delayed = Counter("vllm:prefill_delayed_requests_by_reason_total", "Request steps", ["reason"], registry=registry)
    blocked = Counter("vllm:prefill_blocked_requests_by_reason_total", "Request steps", ["reason"], registry=registry)
    for reason in ("kv_lookup_pending", "kv_load_pending", "mamba_alignment"):
        delayed.labels(reason).inc(0)
        blocked.labels(reason).inc(0)
    collector.parse("replica0", generate_latest(registry).decode(), 10)
    for name, value in {
        "kv_lookup_completed": 4,
        "kv_lookup_duration_ns": 8_000_000,
        "kv_lookup_pending_completed": 2,
        "kv_lookup_pending_duration_ns": 7_000_000,
        "delay_steps": 5,
    }.items():
        counters[name].inc(value)
    delayed.labels("kv_lookup_pending").inc(10)
    blocked.labels("kv_lookup_pending").inc(10)
    delayed.labels("mamba_alignment").inc(5)
    blocked.labels("mamba_alignment").inc(1)
    payload = generate_latest(registry).decode()
    result = collector.parse("replica0", payload, 20)
    prefix = "vllm/replica0/"
    assert result[prefix + "prefill_kv_lookup_latency_ms"] == 2
    assert result[prefix + "prefill_kv_lookup_pending_latency_ms"] == 3.5
    assert result[prefix + "prefill_blocked_requests_per_step/reason/kv_lookup_pending"] == 2
    assert result[prefix + "prefill_delayed_requests_per_step/reason/mamba_alignment"] == 1
    assert result[prefix + "prefill_blocked_requests_per_step/reason/mamba_alignment"] == 0.2
    assert result[prefix + "prefill_blocked_requests_per_step/reason/kv_load_pending"] == 0
    idle = collector.parse("replica0", payload, 30)
    assert prefix + "prefill_kv_lookup_latency_ms" not in idle
    assert prefix + "prefill_blocked_requests_per_step/reason/kv_lookup_pending" not in idle


def test_prefill_readiness_preserves_engines_and_pools_interval_observations():
    collector = InferenceMetricsCollector(config())

    def payload(values: list[tuple[int, int, int, int]]) -> str:
        registry = CollectorRegistry()
        names = ("readiness_steps", "ready_requests", "ready_unscheduled_requests", "unexamined_requests")
        counters = [
            Counter(f"vllm:prefill_{name}_total", "Readiness", ["engine"], registry=registry) for name in names
        ]
        for engine, row in enumerate(values):
            for counter, value in zip(counters, row):
                counter.labels(str(engine)).inc(value)
        return generate_latest(registry).decode()

    prefix = "vllm/replica0/prefill_"
    collector.parse("replica0", payload([(0, 0, 0, 0), (0, 0, 0, 0)]), 10)
    values = [(2, 12, 2, 6), (8, 16, 0, 8)]
    result = collector.parse("replica0", payload(values), 20)
    assert result[prefix + "ready_requests_per_step/engine/0"] == pytest.approx(6)
    assert result[prefix + "ready_requests_per_step/engine/1"] == 2
    assert result[prefix + "ready_requests_per_step"] == pytest.approx(2.8)
    assert result[prefix + "ready_unscheduled_requests_per_step/engine/0"] == 1
    assert result[prefix + "ready_unscheduled_requests_per_step/engine/1"] == 0
    assert result[prefix + "unexamined_requests_per_step"] == pytest.approx(1.4)
    idle = collector.parse("replica0", payload(values), 30)
    assert prefix + "ready_requests_per_step" not in idle
    reset = collector.parse("replica0", payload([(0, 0, 0, 0), (10, 22, 0, 10)]), 40)
    assert prefix + "ready_requests_per_step/engine/0" not in reset
    assert reset[prefix + "ready_requests_per_step/engine/1"] == pytest.approx(3)
    assert reset[prefix + "ready_requests_per_step"] == pytest.approx(3)


@pytest.mark.parametrize("metric", metrics_module.PREFILL_OUTCOME_METRICS)
@pytest.mark.parametrize("openmetrics", [False, True])
def test_prefill_outcome_averages_preserve_labels_engines_and_reset_handling(metric: str, openmetrics: bool) -> None:
    collector = InferenceMetricsCollector(config())
    numerator, denominator, label = metrics_module.PREFILL_OUTCOME_METRICS[metric]

    def payload(values: list[tuple[int, int]]) -> str:
        registry = CollectorRegistry()
        steps = Counter(f"vllm:prefill_{denominator}_total", "Steps", ["engine"], registry=registry)
        counts = Counter(f"vllm:prefill_{numerator}_total", "Outcomes", ["engine", label], registry=registry)
        for engine, (step_count, request_count) in enumerate(values):
            steps.labels(str(engine)).inc(step_count)
            counts.labels(str(engine), "mamba_alignment").inc(request_count)
            counts.labels(str(engine), "other").inc(0)
        if openmetrics:
            from prometheus_client.openmetrics.exposition import generate_latest as generate_openmetrics

            return generate_openmetrics(registry, escaping="allow-utf-8").decode()
        return generate_latest(registry).decode()

    def parse(values: list[tuple[int, int]], time: float) -> dict[str, float]:
        return collector.parse(
            "replica0",
            payload(values),
            time,
            content_type="application/openmetrics-text" if openmetrics else "text/plain",
        )

    prefix = f"vllm/replica0/{metric}"
    initial = parse([(0, 0), (0, 0)], 10)
    assert not any(key.startswith(prefix + "/") for key in initial)
    result = parse([(2, 12), (8, 16)], 20)
    assert result[f"{prefix}/engine/0/{label}/mamba_alignment"] == pytest.approx(6)
    assert result[f"{prefix}/engine/1/{label}/mamba_alignment"] == 2
    assert result[f"{prefix}/{label}/mamba_alignment"] == pytest.approx(2.8)
    assert result[f"{prefix}/{label}/other"] == 0
    idle = parse([(2, 12), (8, 16)], 30)
    assert not any(key.startswith(prefix + "/") for key in idle)
    reset = parse([(0, 0), (10, 22)], 40)
    assert f"{prefix}/engine/0/{label}/mamba_alignment" not in reset
    assert reset[f"{prefix}/{label}/mamba_alignment"] == pytest.approx(3)


def test_prefill_lookup_phases_share_completed_attempts_and_use_interval_deltas():
    collector = InferenceMetricsCollector(config())
    values = {
        "kv_lookup_completed": 20,
        "kv_lookup_duration_ns": 200_000_000,
        "kv_lookup_timed_completed": 2,
        "kv_lookup_timed_failed": 1,
        "kv_lookup_queue_duration_ns": 4_000_000,
        "kv_lookup_execution_duration_ns": 6_000_000,
        "kv_lookup_observation_duration_ns": 10_000_000,
    }

    def payload(counts):
        registry = CollectorRegistry()
        for name, value in counts.items():
            Counter(f"vllm:prefill_{name}_total", "Lookup timing", registry=registry).inc(value)
        return generate_latest(registry).decode()

    prefix = "vllm/replica0/prefill_kv_lookup_"
    derived = {
        prefix + phase + "_latency_ms": value
        for phase, value in (("queue", 2), ("execution", 3), ("observation", 5), ("client", 10))
    }
    initial = collector.parse("replica0", payload(dict.fromkeys(values, 0)), 10)
    assert not derived.keys() & initial.keys()
    result = collector.parse("replica0", payload(values), 20)
    for key, value in derived.items():
        assert result[key] == value
    assert result[prefix + "timed_failed_per_second"] == 0.1
    idle = collector.parse("replica0", payload(values), 30)
    assert not derived.keys() & idle.keys()
    reset = collector.parse("replica0", payload(dict.fromkeys(values, 0)), 40)
    assert not derived.keys() & reset.keys()
    # An older/partial exporter must not produce a misleading summed latency.
    values.pop("kv_lookup_observation_duration_ns")
    partial = collector.parse("replica0", payload(values), 50)
    assert partial[prefix + "queue_latency_ms"] == 2
    assert prefix + "client_latency_ms" not in partial
