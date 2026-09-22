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
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from omegaconf import DictConfig
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


def test_counter_rates_reset_and_nonfinite():
    collector = InferenceMetricsCollector(config())

    def sample(value, now):
        return collector.parse(
            "replica0", f"# TYPE vllm:generation_tokens_total counter\nvllm:generation_tokens_total {value}\n", now
        )

    rate = "vllm/replica0/generation_tokens_per_second"
    assert rate not in sample(100, 10)
    assert sample(160, 12)[rate] == 30
    assert rate not in sample(5, 14)
    assert sample(25, 16)[rate] == 10
    assert sample("NaN", 17) == {}


async def test_real_http_scrape_publishes(monkeypatch):
    app = web.Application()

    async def serve(request):
        return web.Response(text=GAUGES)

    app.router.add_get("/metrics", serve)
    publish = MagicMock()
    monkeypatch.setattr(metrics_module, "export_metrics", publish)
    async with TestServer(app) as server, ClientSession() as session:
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", session)
        cfg = InferenceMetricsConfig(enabled=True, endpoints={"replica0": str(server.make_url("/metrics"))})
        await InferenceMetricsCollector(cfg).scrape("replica0", cfg.endpoints["replica0"])
    payload = publish.call_args.args[0]
    assert payload["vllm/replica0/num_requests_running"] == 7
    assert all(key.startswith("vllm/replica0/") for key in payload)
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
        assert not any("only_on_" in key for key in payload)
        assert len(payload) == 8


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
