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

"""Reproducible connection-pool telemetry benchmark for issue #3008."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from aiohttp import TraceConfig, web

from nemo_gym import server_utils
from nemo_gym.telemetry._fallbacks import safe_set_span_attributes


_SPAN_PROVIDER = None
_SPAN_EXPORTER = None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _open_resource_counts() -> tuple[int, int]:
    fd_dir = Path("/proc/self/fd")
    try:
        entries = list(fd_dir.iterdir())
    except OSError:
        return -1, -1
    sockets = 0
    for entry in entries:
        try:
            sockets += entry.readlink().as_posix().startswith("socket:")
        except OSError:
            continue
    return len(entries), sockets


def _recording_telemetry():
    """Install one in-memory recording SDK provider per spawned worker process."""
    global _SPAN_EXPORTER, _SPAN_PROVIDER
    if _SPAN_PROVIDER is None:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        _SPAN_EXPORTER = InMemorySpanExporter()
        _SPAN_PROVIDER = TracerProvider()
        _SPAN_PROVIDER.add_span_processor(SimpleSpanProcessor(_SPAN_EXPORTER))
        trace.set_tracer_provider(_SPAN_PROVIDER)
    _SPAN_EXPORTER.clear()
    return _SPAN_PROVIDER, _SPAN_EXPORTER


def _measurement_trace(queue_samples_ms: list[float]) -> TraceConfig:
    def factory(*, trace_request_ctx=None):
        return SimpleNamespace(started_at=None, duration_ms=0.0, recorded=False)

    trace = TraceConfig(trace_config_ctx_factory=factory)

    async def queue_start(_session, context, _params):
        if context.started_at is None:
            context.started_at = time.perf_counter()

    async def queue_end(_session, context, _params):
        if context.started_at is not None:
            context.duration_ms += (time.perf_counter() - context.started_at) * 1000.0
            context.started_at = None

    def record(context) -> None:
        if context.recorded:
            return
        if context.started_at is not None:
            context.duration_ms += (time.perf_counter() - context.started_at) * 1000.0
        queue_samples_ms.append(context.duration_ms)
        context.recorded = True

    async def request_end(_session, context, _params):
        record(context)

    async def request_exception(_session, context, _params):
        record(context)

    trace.on_connection_queued_start.append(queue_start)
    trace.on_connection_queued_end.append(queue_end)
    trace.on_request_end.append(request_end)
    trace.on_request_exception.append(request_exception)
    return trace


@contextmanager
def _baseline_client_span(method: str, url: str):
    """Match main's existing CLIENT span without the candidate queue context."""
    from nemo_gym.telemetry.contrib import inject_trace_context
    from nemo_gym.telemetry.spans import client_span

    with client_span(f"HTTP {method.upper()}") as span:
        headers = {}
        inject_trace_context(headers)
        safe_set_span_attributes(
            span,
            {"http.request.method": method.upper(), "url.full": server_utils._redacted_url(url)},
        )
        yield span, headers


async def _baseline_request(method: str, url: str):
    with _baseline_client_span(method, url) as (span, headers):
        response = await server_utils._request_with_retries(method, url, _max_connection_retries=1, headers=headers)
        safe_set_span_attributes(span, {"http.response.status_code": response.status})
        return response


def _backend_main(ports: list[int], delay_seconds: float, ready, stop) -> None:
    async def run() -> None:
        runners = []

        async def fixed_latency(_request):
            started_at = time.perf_counter()
            await asyncio.sleep(delay_seconds)
            service_ms = (time.perf_counter() - started_at) * 1000.0
            return web.Response(body=b"ok", headers={"X-Backend-Service-Ms": str(service_ms)})

        for port in ports:
            app = web.Application()
            app.router.add_get("/work", fixed_latency)
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            await web.TCPSite(runner, "127.0.0.1", port).start()
            runners.append(runner)
        ready.set()
        try:
            while not stop.is_set():
                await asyncio.sleep(0.05)
        finally:
            for runner in runners:
                await runner.cleanup()

    asyncio.run(run())


def _free_ports(count: int) -> list[int]:
    import socket

    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


async def _run_worker_async(spec: dict) -> dict:
    queue_probe_ms: list[float] = []

    cfg = server_utils.GlobalAIOHTTPAsyncClientConfig(
        global_aiohttp_connector_limit=spec["aggregate_limit"],
        global_aiohttp_connector_limit_per_host=spec["aggregate_limit_per_host"],
    )
    capacity = server_utils._connection_pool_capacity(cfg, spec["workers"])
    provider = exporter = None
    if spec["mode"] != "telemetry_off":
        provider, exporter = _recording_telemetry()

    old_client = server_utils._GLOBAL_AIOHTTP_CLIENT
    old_gate = server_utils.is_span_group_enabled
    old_workers = server_utils.get_nemo_gym_fastapi_num_workers
    errors = Counter()
    client_call_ms = []
    full_response_ms = []
    service_ms = []
    sampling = True

    # Build the session through the production entrypoint so the benchmark exercises the real
    # connector configuration (keepalive timeout, keepalive socket factory, DummyCookieJar,
    # ClientTimeout) and the startup capacity report, rather than a bare ClientSession.
    server_utils.is_span_group_enabled = lambda _group: spec["mode"] == "telemetry_on"
    server_utils.get_nemo_gym_fastapi_num_workers = lambda: spec["workers"]
    server_utils._GLOBAL_AIOHTTP_CLIENT = None
    session = server_utils.set_global_aiohttp_client(cfg)
    if spec["mode"] == "baseline_queue_probe":
        # aiohttp reads _trace_configs per request, so appending after construction is honoured,
        # but ClientSession freezes the configs it was built with and refuses to dispatch a
        # non-frozen signal, so the late addition has to be frozen explicitly.
        probe = _measurement_trace(queue_probe_ms)
        probe.freeze()
        session._trace_configs.append(probe)

    async with session:

        async def perform(index: int, *, record: bool) -> None:
            started_at = time.perf_counter()
            try:
                url = spec["urls"][index % spec["destinations"]]
                if spec["mode"].startswith("baseline"):
                    response = await _baseline_request("GET", url)
                else:
                    response = await server_utils.request("GET", url, _max_connection_retries=1)
                if record:
                    client_call_ms.append((time.perf_counter() - started_at) * 1000.0)
                    service_ms.append(float(response.headers["X-Backend-Service-Ms"]))
                await response.read()
                if response.status != 200:
                    errors[f"http_{response.status}"] += 1
            except Exception as error:
                errors[type(error).__name__] += 1
            finally:
                if record:
                    full_response_ms.append((time.perf_counter() - started_at) * 1000.0)

        async def drive(count: int, *, record: bool) -> None:
            semaphore = asyncio.Semaphore(spec["concurrency_per_worker"])

            async def bounded(index: int) -> None:
                async with semaphore:
                    await perform(index, record=record)

            await asyncio.gather(*(bounded(index) for index in range(count)))

        try:
            await drive(spec["warmup_requests"], record=False)
            queue_probe_ms.clear()
            if exporter is not None:
                exporter.clear()

            initial_fds, initial_sockets = _open_resource_counts()
            if initial_fds < 0 or initial_sockets < 0:
                raise RuntimeError("FD/socket resource sampling is unavailable")
            peak_fds, peak_sockets = initial_fds, initial_sockets

            async def sample_resources() -> None:
                nonlocal peak_fds, peak_sockets
                while sampling:
                    fds, sockets = _open_resource_counts()
                    peak_fds = max(peak_fds, fds)
                    peak_sockets = max(peak_sockets, sockets)
                    await asyncio.sleep(0.001)

            sampler = asyncio.create_task(sample_resources())
            started_at = time.monotonic()
            try:
                await drive(spec["requests"], record=True)
            finally:
                ended_at = time.monotonic()
                sampling = False
                await sampler
        finally:
            server_utils._GLOBAL_AIOHTTP_CLIENT = old_client
            server_utils.is_span_group_enabled = old_gate
            server_utils.get_nemo_gym_fastapi_num_workers = old_workers

    spans = []
    if provider is not None:
        provider.force_flush()
        spans = [span for span in exporter.get_finished_spans() if span.kind.name == "CLIENT"]
    span_ms = [(span.end_time - span.start_time) / 1_000_000 for span in spans]
    if spec["mode"] == "telemetry_on":
        queue_ms = [span.attributes["nemo.gym.http.connection_pool.queue_duration_ms"] for span in spans]
    else:
        queue_ms = queue_probe_ms

    expected = spec["requests"]
    # Errors are a recorded measurement, not a fatal condition. The undersized cells are exactly
    # where timeouts and connection errors appear, so aborting there discarded the data the
    # benchmark plan asks for. Every attempted request must still be accounted for.
    if len(full_response_ms) != expected:
        raise RuntimeError("Benchmark request sample count does not match the requested count")
    if len(client_call_ms) > expected or len(service_ms) > expected:
        raise RuntimeError("Benchmark recorded more samples than requests")
    if spec["mode"] != "telemetry_off" and len(spans) != expected:
        raise RuntimeError("Exported CLIENT span count does not match the requested count")
    if spec["mode"] in {"baseline_queue_probe", "telemetry_on"} and len(queue_ms) > expected:
        raise RuntimeError("Queue sample count exceeds the requested count")
    end_fds, end_sockets = _open_resource_counts()
    if end_fds < 0 or end_sockets < 0:
        raise RuntimeError("FD/socket resource sampling is unavailable")

    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "queue_ms": queue_ms,
        "client_span_ms": span_ms,
        "client_call_ms": client_call_ms,
        "full_response_ms": full_response_ms,
        "backend_service_ms": service_ms,
        "initial_fds": initial_fds,
        "initial_sockets": initial_sockets,
        "peak_fds": peak_fds,
        "peak_sockets": peak_sockets,
        "end_fds": end_fds,
        "end_sockets": end_sockets,
        "errors": dict(errors),
        "effective_limit": capacity.total,
        "effective_limit_per_host": capacity.per_host,
    }


def _run_worker(spec: dict) -> dict:
    return asyncio.run(_run_worker_async(spec))


def _aggregate(worker_results: list[dict], requests: int) -> dict:
    raw_fields = ("queue_ms", "client_span_ms", "client_call_ms", "full_response_ms", "backend_service_ms")
    raw = {field: [value for result in worker_results for value in result[field]] for field in raw_fields}
    errors = Counter()
    for result in worker_results:
        errors.update(result["errors"])
    wall_seconds = max(result["ended_at"] for result in worker_results) - min(
        result["started_at"] for result in worker_results
    )
    total_requests = requests * len(worker_results)
    error_count = sum(errors.values())
    return {
        "metrics": {field: _summary(values) for field, values in raw.items()},
        "queued_requests": sum(value > 0 for value in raw["queue_ms"]),
        "queued_only_ms": _summary([value for value in raw["queue_ms"] if value > 0]),
        "throughput_requests_per_second": total_requests / wall_seconds,
        "error_count": error_count,
        "error_rate": error_count / total_requests,
        "aggregate_initial_fds": sum(result["initial_fds"] for result in worker_results),
        "aggregate_initial_sockets": sum(result["initial_sockets"] for result in worker_results),
        "aggregate_peak_fds": sum(result["peak_fds"] for result in worker_results),
        "aggregate_peak_sockets": sum(result["peak_sockets"] for result in worker_results),
        "aggregate_end_fds": sum(result["end_fds"] for result in worker_results),
        "aggregate_end_sockets": sum(result["end_sockets"] for result in worker_results),
        "aggregate_peak_fd_delta": sum(result["peak_fds"] - result["initial_fds"] for result in worker_results),
        "aggregate_peak_socket_delta": sum(
            result["peak_sockets"] - result["initial_sockets"] for result in worker_results
        ),
        "max_peak_fds_per_worker": max(result["peak_fds"] for result in worker_results),
        "max_peak_sockets_per_worker": max(result["peak_sockets"] for result in worker_results),
        "errors": dict(errors),
        "raw": raw,
    }


def _limits(scenario: str, aggregate_concurrency: int, workers: int) -> tuple[int, int, int]:
    if scenario == "undersized_per_host":
        return aggregate_concurrency, max(workers, aggregate_concurrency // 4), 1
    if scenario == "undersized_total":
        return max(workers, aggregate_concurrency // 4), aggregate_concurrency, 4
    return aggregate_concurrency * 2, aggregate_concurrency * 2, 4


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for package in ("aiohttp", "nemo-lens", "opentelemetry-api", "opentelemetry-sdk"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _host_metadata() -> dict:
    cpu_model = "unknown"
    memory = "unknown"
    physical_cores = set()
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
        for processor in cpuinfo.split("\n\n"):
            values = dict(
                (key.strip(), value.strip())
                for line in processor.splitlines()
                if ":" in line
                for key, value in [line.split(":", 1)]
            )
            physical_id = values.get("physical id")
            core_id = values.get("core id")
            if physical_id is not None and core_id is not None:
                physical_cores.add((physical_id.strip(), core_id.strip()))
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                memory = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    lock = Path("uv.lock")
    return {
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "physical_cores": len(physical_cores) or "unknown",
        "logical_cpus": os.cpu_count(),
        "memory": memory,
        "rlimit_nofile": resource.getrlimit(resource.RLIMIT_NOFILE),
        "ephemeral_ports_per_destination": server_utils._ephemeral_port_capacity(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    }


def _variance(records: list[dict]) -> list[dict]:
    grouped = {}
    for record in records:
        key = (record["workers"], record["aggregate_concurrency"], record["scenario"], record["mode"])
        grouped.setdefault(key, []).append(record)
    results = []
    for key, group in grouped.items():
        throughput = [record["throughput_requests_per_second"] for record in group]
        metric_variance = {}
        for metric in ("queue_ms", "client_span_ms", "client_call_ms", "backend_service_ms"):
            metric_variance[metric] = {}
            for percentile in ("p50", "p95", "p99"):
                values = [record["metrics"][metric][percentile] for record in group]
                metric_variance[metric][f"{percentile}_median"] = statistics.median(values)
                metric_variance[metric][f"{percentile}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        results.append(
            {
                "workers": key[0],
                "aggregate_concurrency": key[1],
                "scenario": key[2],
                "mode": key[3],
                "throughput_median": statistics.median(throughput),
                "throughput_stdev": statistics.stdev(throughput) if len(throughput) > 1 else 0.0,
                "throughput_cv": statistics.stdev(throughput) / statistics.mean(throughput)
                if len(throughput) > 1
                else 0.0,
                "metrics": metric_variance,
                "aggregate_peak_fds_median": statistics.median(record["aggregate_peak_fds"] for record in group),
                "aggregate_peak_sockets_median": statistics.median(
                    record["aggregate_peak_sockets"] for record in group
                ),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default="1,4,16")
    parser.add_argument("--concurrency", default="128,256,512", help="Aggregate concurrency across workers")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--requests-per-worker", type=int, default=256)
    parser.add_argument("--delay-ms", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    worker_counts = [int(value) for value in args.workers.split(",")]
    concurrencies = [int(value) for value in args.concurrency.split(",")]
    modes = ["baseline", "baseline_queue_probe", "telemetry_off", "telemetry_on"]
    scenarios = ["undersized_per_host", "undersized_total", "adequate"]
    context = multiprocessing.get_context("spawn")
    ports = _free_ports(4)
    ready = context.Event()
    stop = context.Event()
    backend = context.Process(target=_backend_main, args=(ports, args.delay_ms / 1000.0, ready, stop))
    backend.start()
    if not ready.wait(timeout=30):
        backend.terminate()
        backend.join()
        raise RuntimeError("Loopback backend did not start")
    urls = [f"http://127.0.0.1:{port}/work" for port in ports]
    records = []

    try:
        for workers in worker_counts:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                for aggregate_concurrency in concurrencies:
                    concurrency_per_worker = max(1, aggregate_concurrency // workers)
                    for scenario in scenarios:
                        aggregate_limit, aggregate_limit_per_host, destinations = _limits(
                            scenario, aggregate_concurrency, workers
                        )
                        for repetition in range(args.repetitions):
                            ordered_modes = modes if repetition % 2 == 0 else list(reversed(modes))
                            for mode in ordered_modes:
                                spec = {
                                    "mode": mode,
                                    "workers": workers,
                                    "aggregate_limit": aggregate_limit,
                                    "aggregate_limit_per_host": aggregate_limit_per_host,
                                    "concurrency_per_worker": concurrency_per_worker,
                                    "destinations": destinations,
                                    "urls": urls,
                                    "requests": max(args.requests_per_worker, concurrency_per_worker * 10),
                                    "warmup_requests": max(concurrency_per_worker, 8),
                                }
                                worker_results = [
                                    future.result()
                                    for future in [executor.submit(_run_worker, spec) for _ in range(workers)]
                                ]
                                aggregate = _aggregate(worker_results, spec["requests"])
                                records.append(
                                    {
                                        "workers": workers,
                                        "aggregate_concurrency": aggregate_concurrency,
                                        "concurrency_per_worker": concurrency_per_worker,
                                        "requests_per_worker": spec["requests"],
                                        "scenario": scenario,
                                        "aggregate_limit": aggregate_limit,
                                        "aggregate_limit_per_host": aggregate_limit_per_host,
                                        "effective_limit_per_worker": worker_results[0]["effective_limit"],
                                        "effective_limit_per_host_per_worker": worker_results[0][
                                            "effective_limit_per_host"
                                        ],
                                        "realized_aggregate_limit": worker_results[0]["effective_limit"] * workers,
                                        "realized_aggregate_limit_per_host": worker_results[0][
                                            "effective_limit_per_host"
                                        ]
                                        * workers,
                                        "mode": mode,
                                        "repetition": repetition,
                                        **aggregate,
                                    }
                                )
                                print(
                                    f"workers={workers} aggregate_concurrency={aggregate_concurrency} "
                                    f"scenario={scenario} mode={mode} repetition={repetition}",
                                    flush=True,
                                )
    finally:
        stop.set()
        backend.join(timeout=30)
        if backend.is_alive():
            backend.terminate()
            backend.join()

    output = {
        "metadata": {
            "command": [sys.executable, *sys.argv],
            "python": platform.python_version(),
            "dependencies": _dependency_versions(),
            "host": _host_metadata(),
            "delay_ms": args.delay_ms,
            "requests_per_worker": args.requests_per_worker,
            "repetitions": args.repetitions,
            "worker_counts": worker_counts,
            "aggregate_concurrencies": concurrencies,
            "telemetry_export": "OpenTelemetry SDK SimpleSpanProcessor with InMemorySpanExporter",
            "modes": {
                "baseline": "Current-main CLIENT span without queue instrumentation",
                "baseline_queue_probe": "Current-main CLIENT span plus observer-only queue timing",
                "telemetry_off": "Production disabled path with no TraceConfig or CLIENT span",
                "telemetry_on": "Candidate CLIENT span and connection-pool queue instrumentation",
            },
        },
        "records": records,
        "cross_repetition_summary": _variance(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
