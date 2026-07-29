# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context-scoped recorder for NeMo Gym observability.

Spans are recorded to JSONL on disk and emitted as OTel traces via any
configured TracerProvider (e.g. NeMo Lens wired to Honeycomb / Grafana Tempo).
When no TracerProvider is configured the OTel calls are zero-cost no-ops.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Iterator

from nemo_gym.observability.events import (
    SCHEMA_VERSION,
    ResourceSample,
    SandboxEvent,
    safe_attributes,
    stable_hash,
)
from nemo_gym.observability.summary import (
    flatten_summary_metrics,
    load_jsonl,
    write_summary,
)
from nemo_gym.observability.traces import export_trace_artifacts


G_CURRENT_RECORDER: ContextVar[SandboxEventRecorder | None] = ContextVar(
    "nemo_gym_observability_recorder",
    default=None,
)
G_EVENT_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "nemo_gym_observability_context",
    default={},
)
G_SUPPRESS_EVENTS: ContextVar[int] = ContextVar(
    "nemo_gym_observability_suppress",
    default=0,
)
G_SPAN_STACK: ContextVar[tuple[dict[str, str], ...]] = ContextVar(
    "nemo_gym_observability_span_stack",
    default=(),
)
G_ENV_RECORDER: SandboxEventRecorder | None = None
G_ENV_RECORDER_LOCK = threading.Lock()
_METRIC_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@contextmanager
def _otel_span_cm(name: str, attributes: dict[str, Any]) -> Iterator[None]:
    """Emit a best-effort OTel trace span alongside JSONL recording.

    No-op if opentelemetry-api is not installed or no TracerProvider is configured
    (e.g. NeMo Lens has not been initialised via setup_telemetry()).
    """
    try:
        import opentelemetry.trace as _ot
    except ImportError:
        yield
        return

    otel_attrs: dict[str, Any] = {}
    for k, v in attributes.items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float)):
            otel_attrs[k] = v
        elif isinstance(v, str):
            otel_attrs[k] = v
        else:
            otel_attrs[k] = str(v)

    tracer = _ot.get_tracer("nemo_gym.observability")
    with tracer.start_as_current_span(name, attributes=otel_attrs) as span:
        try:
            yield
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(_ot.StatusCode.ERROR, str(exc))
            raise


class SandboxEventRecorder:
    """Append-only event recorder with JSONL, OTel trace, and W&B finalization."""

    def __init__(
        self,
        *,
        output_dir: Path,
        resource_sample_interval_s: float,
        max_rendered_trajectories: int,
        artifacts: dict[str, Any],
        otel: dict[str, Any],
        wandb: dict[str, Any],
        process_trace: dict[str, Any],
        privacy: dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.events_path = output_dir / "events.jsonl"
        self.resource_samples_path = output_dir / "resource_samples.jsonl"
        self.resource_sample_interval_s = resource_sample_interval_s
        self.max_rendered_trajectories = max_rendered_trajectories
        self.artifacts = artifacts
        self.otel = otel
        self.wandb = wandb
        self.process_trace = process_trace
        self.privacy = privacy
        self.run_id = run_id
        self.include_command_text = bool(privacy["include_command_text"])
        self._start_wall_time_s = time.time()
        self._start_monotonic_s = time.monotonic()
        self._lock = threading.Lock()
        self._closed = False
        self._otel_sink = _OtelSink(otel)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_event(
            "lifecycle",
            "run.start",
            attributes={"run_id": run_id},
        )

    def record_event(
        self,
        event_type: str,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        timestamp_unix_s: float | None = None,
        monotonic_s: float | None = None,
    ) -> SandboxEvent:
        """Append one event and mirror it to configured live sinks."""
        attrs = safe_attributes({**G_EVENT_CONTEXT.get(), **(attributes or {})})
        event_time = time.time() if timestamp_unix_s is None else timestamp_unix_s
        event_monotonic = time.monotonic() if timestamp_unix_s is None and monotonic_s is None else monotonic_s
        event: SandboxEvent = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_unix_s": event_time,
            "monotonic_s": event_monotonic,
            "elapsed_time_s": self._elapsed_time_s(
                timestamp_unix_s=event_time,
                monotonic_s=event_monotonic,
            ),
            "event_type": event_type,
            "name": name,
            "attributes": attrs,
        }
        if trace_id:
            event["trace_id"] = trace_id
        if span_id:
            event["span_id"] = span_id
        if parent_span_id:
            event["parent_span_id"] = parent_span_id
        self._append_jsonl(self.events_path, event)
        self._otel_sink.record_event(event)
        return event

    def record_resource_sample(
        self,
        *,
        sandbox_id: str | None,
        sample: dict[str, Any],
        attributes: dict[str, Any] | None = None,
    ) -> ResourceSample:
        """Append one resource sample."""
        attrs = safe_attributes({**G_EVENT_CONTEXT.get(), **(attributes or {})})
        sample_time = time.time()
        sample_monotonic = time.monotonic()
        row: ResourceSample = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_unix_s": sample_time,
            "monotonic_s": sample_monotonic,
            "elapsed_time_s": self._elapsed_time_s(
                timestamp_unix_s=sample_time,
                monotonic_s=sample_monotonic,
            ),
            "sandbox_id": sandbox_id,
            "attributes": attrs,
        }
        for key in (
            "cpu_utilization",
            "cpu_usage_seconds_total",
            "memory_usage_bytes",
            "process_count",
            "processes",
        ):
            if key in sample:
                row[key] = sample[key]
        self._append_jsonl(self.resource_samples_path, row)
        self._otel_sink.record_resource_sample(row)
        return row

    def resource_sampler_interval_s(self) -> float:
        """Return the interval needed by resource and process sampling."""
        intervals = []
        if self.resource_sample_interval_s > 0:
            intervals.append(self.resource_sample_interval_s)
        if self.process_trace.get("enabled"):
            interval_s = float(self.process_trace["sample_interval_s"])
            if interval_s > 0:
                intervals.append(interval_s)
        return min(intervals) if intervals else 0.0

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        phase: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Record an async span as start/end events and an OTel trace span."""
        start_monotonic = time.monotonic()
        span_attrs = {"phase": phase, **(attributes or {})}
        span_context = self._new_span_context(name, span_attrs)
        token = G_SPAN_STACK.set(G_SPAN_STACK.get() + (span_context,))
        self.record_event(
            "span_start",
            name,
            attributes=span_attrs,
            trace_id=span_context["trace_id"],
            span_id=span_context["span_id"],
            parent_span_id=span_context.get("parent_span_id"),
        )
        try:
            with _otel_span_cm(name, span_attrs):
                try:
                    yield
                except Exception as e:
                    self.record_event(
                        "span_end",
                        name,
                        attributes={
                            **span_attrs,
                            "status": "error",
                            "duration_s": time.monotonic() - start_monotonic,
                            "error_type": type(e).__name__,
                        },
                        trace_id=span_context["trace_id"],
                        span_id=span_context["span_id"],
                        parent_span_id=span_context.get("parent_span_id"),
                    )
                    raise
                else:
                    self.record_event(
                        "span_end",
                        name,
                        attributes={
                            **span_attrs,
                            "status": "ok",
                            "duration_s": time.monotonic() - start_monotonic,
                        },
                        trace_id=span_context["trace_id"],
                        span_id=span_context["span_id"],
                        parent_span_id=span_context.get("parent_span_id"),
                    )
        finally:
            G_SPAN_STACK.reset(token)

    @contextmanager
    def sync_span(
        self,
        name: str,
        *,
        phase: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Record a synchronous span as start/end events and an OTel trace span."""
        start_monotonic = time.monotonic()
        span_attrs = {"phase": phase, **(attributes or {})}
        span_context = self._new_span_context(name, span_attrs)
        token = G_SPAN_STACK.set(G_SPAN_STACK.get() + (span_context,))
        self.record_event(
            "span_start",
            name,
            attributes=span_attrs,
            trace_id=span_context["trace_id"],
            span_id=span_context["span_id"],
            parent_span_id=span_context.get("parent_span_id"),
        )
        try:
            with _otel_span_cm(name, span_attrs):
                try:
                    yield
                except Exception as e:
                    self.record_event(
                        "span_end",
                        name,
                        attributes={
                            **span_attrs,
                            "status": "error",
                            "duration_s": time.monotonic() - start_monotonic,
                            "error_type": type(e).__name__,
                        },
                        trace_id=span_context["trace_id"],
                        span_id=span_context["span_id"],
                        parent_span_id=span_context.get("parent_span_id"),
                    )
                    raise
                else:
                    self.record_event(
                        "span_end",
                        name,
                        attributes={
                            **span_attrs,
                            "status": "ok",
                            "duration_s": time.monotonic() - start_monotonic,
                        },
                        trace_id=span_context["trace_id"],
                        span_id=span_context["span_id"],
                        parent_span_id=span_context.get("parent_span_id"),
                    )
        finally:
            G_SPAN_STACK.reset(token)

    def ingest_jsonl(
        self,
        path: Path,
        *,
        trajectory_id: str | None = None,
        source: str,
    ) -> int:
        """Ingest externally produced observability events into this run."""
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                attrs = dict(record.get("attributes") or {})
                attrs["source"] = source
                if trajectory_id is not None:
                    attrs.setdefault("trajectory_id", trajectory_id)
                self.record_event(
                    str(record.get("event_type") or "event"),
                    str(record.get("name") or "external.event"),
                    attributes=attrs,
                    timestamp_unix_s=record.get("timestamp_unix_s")
                    if isinstance(record.get("timestamp_unix_s"), (int, float))
                    else None,
                    monotonic_s=None,
                )
                count += 1
        return count

    def finalize(self) -> None:
        """Flush summary, static reports, and best-effort external sinks."""
        if self._closed:
            return
        self._closed = True
        self.record_event("lifecycle", "run.end", attributes={"run_id": self.run_id})
        try:
            write_summary(self.output_dir)
        except OSError as e:
            self.record_event(
                "error",
                "observability.summary_error",
                attributes={"error_type": type(e).__name__, "error": str(e)},
            )
        if self.artifacts.get("export_otlp_json", True):
            try:
                export_trace_artifacts(
                    self.output_dir,
                    service_name=str(self.otel.get("service_name") or "nemo-gym-eval"),
                    run_id=self.run_id,
                )
            except Exception as e:
                self.record_event(
                    "error",
                    "observability.trace_export_error",
                    attributes={"error_type": type(e).__name__, "error": str(e)},
                )
        if self.artifacts["enabled"]:
            try:
                from nemo_gym.observability.render import render_reports

                render_reports(
                    self.output_dir,
                    max_rendered_trajectories=self.max_rendered_trajectories,
                    render_html=bool(self.artifacts["render_html"]),
                    render_png=bool(self.artifacts["render_png"]),
                )
            except Exception as e:  # Best-effort visualization must not fail evals.
                self.record_event(
                    "error",
                    "observability.render_error",
                    attributes={"error_type": type(e).__name__, "error": str(e)},
                )
        try:
            log_wandb_artifact(self.output_dir, self.wandb)
        finally:
            self._otel_sink.shutdown()

    def _append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")

    def _elapsed_time_s(
        self,
        *,
        timestamp_unix_s: float,
        monotonic_s: float | None,
    ) -> float:
        if monotonic_s is not None:
            return max(0.0, float(monotonic_s) - self._start_monotonic_s)
        return max(0.0, float(timestamp_unix_s) - self._start_wall_time_s)

    def _new_span_context(self, name: str, attrs: dict[str, Any]) -> dict[str, str]:
        parent = G_SPAN_STACK.get()[-1] if G_SPAN_STACK.get() else None
        trace_seed = (
            parent["trace_id"]
            if parent
            else ":".join(
                [
                    self.run_id or "run",
                    str(
                        attrs.get("trajectory_id")
                        or attrs.get("trial_name")
                        or G_EVENT_CONTEXT.get().get("trajectory_id")
                        or G_EVENT_CONTEXT.get().get("trial_name")
                        or attrs.get("sandbox_id")
                        or name
                    ),
                ]
            )
        )
        trace_id = trace_seed if parent else stable_hash(trace_seed, length=32)
        span_id = stable_hash(
            f"{trace_id}:{name}:{time.time_ns()}:{threading.get_ident()}",
            length=16,
        )
        context = {
            "trace_id": trace_id,
            "span_id": span_id,
        }
        if parent:
            context["parent_span_id"] = parent["span_id"]
        return context


class _OtelSink:
    """Best-effort OpenTelemetry metrics sink (histograms + counters via OTLP HTTP)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._meter_provider = None
        self._duration_histogram = None
        self._phase_duration_histograms = {}
        self._counter = None
        self._memory_histogram = None
        self._cpu_histogram = None
        if not cfg["enabled"]:
            return
        endpoint = cfg["endpoint"]
        if not endpoint:
            return
        try:
            from opentelemetry import metrics
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource
        except ImportError:
            return
        exporter = OTLPMetricExporter(endpoint=endpoint)
        reader = PeriodicExportingMetricReader(exporter)
        self._meter_provider = MeterProvider(
            metric_readers=[reader],
            resource=Resource.create({"service.name": cfg["service_name"]}),
        )
        metrics.set_meter_provider(self._meter_provider)
        meter = metrics.get_meter("nemo_gym.observability")
        self._duration_histogram = meter.create_histogram(
            "nemo_gym.operation.duration",
            unit="s",
            description="NeMo Gym operation duration.",
        )
        self._phase_duration_histograms = {
            "startup": meter.create_histogram(
                "nemo_gym.startup.duration",
                unit="s",
                description="Sandbox startup duration.",
            ),
            "setup": meter.create_histogram(
                "nemo_gym.setup.duration",
                unit="s",
                description="Sandbox setup duration.",
            ),
            "execution": meter.create_histogram(
                "nemo_gym.execution.duration",
                unit="s",
                description="Sandbox execution duration.",
            ),
            "llm": meter.create_histogram(
                "nemo_gym.llm.request.duration",
                unit="s",
                description="LLM request duration.",
            ),
        }
        self._counter = meter.create_counter(
            "nemo_gym.events",
            description="NeMo Gym event counts.",
        )
        self._memory_histogram = meter.create_histogram(
            "nemo_gym.resource.memory.usage_bytes",
            unit="By",
            description="Sandbox memory usage samples.",
        )
        self._cpu_histogram = meter.create_histogram(
            "nemo_gym.resource.cpu.utilization",
            unit="1",
            description="Sandbox CPU utilization samples.",
        )

    def record_event(self, event: SandboxEvent) -> None:
        attrs = event.get("attributes") or {}
        otel_attrs = _low_cardinality_attrs(attrs)
        otel_attrs["event_name"] = event["name"]
        if self._counter is not None:
            self._counter.add(1, otel_attrs)
        duration_s = attrs.get("duration_s")
        if self._duration_histogram is not None and isinstance(duration_s, (int, float)):
            self._duration_histogram.record(float(duration_s), otel_attrs)
            phase = str(attrs.get("phase") or "")
            phase_histogram = self._phase_duration_histograms.get(phase)
            if phase_histogram is not None:
                phase_histogram.record(float(duration_s), otel_attrs)

    def record_resource_sample(self, sample: ResourceSample) -> None:
        attrs = _low_cardinality_attrs(sample.get("attributes") or {})
        if self._memory_histogram is not None and isinstance(sample.get("memory_usage_bytes"), int):
            self._memory_histogram.record(float(sample["memory_usage_bytes"]), attrs)
        if self._cpu_histogram is not None and isinstance(sample.get("cpu_utilization"), (int, float)):
            self._cpu_histogram.record(float(sample["cpu_utilization"]), attrs)

    def shutdown(self) -> None:
        if self._meter_provider is not None:
            self._meter_provider.shutdown()


def _low_cardinality_attrs(attributes: dict[str, Any]) -> dict[str, str]:
    allowed = (
        "phase",
        "status",
        "provider",
        "batch_mode",
        "command_class",
        "stop_reason",
        "harness",
        "source",
    )
    return {key: str(attributes[key]) for key in allowed if key in attributes and attributes[key] is not None}


def _safe_metric_component(value: Any) -> str:
    cleaned = _METRIC_NAME_RE.sub("_", str(value))[:80].strip("._-")
    return cleaned or "unknown"


def log_wandb_artifact(output_dir: Path, cfg: dict[str, Any]) -> None:
    """Log observability metrics and artifacts to W&B when configured."""
    enabled = cfg.get("enabled")
    if enabled is False:
        return
    try:
        import wandb
    except ImportError:
        return
    created_run = False
    if wandb.run is None:
        if enabled is not True:
            return
        wandb_dir = Path(os.environ.get("WANDB_DIR", output_dir.parent / "wandb"))
        wandb_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs = {
            "project": cfg.get("project") or os.environ.get("WANDB_PROJECT") or "nemo-gym-eval",
            "name": cfg.get("run_name") or os.environ.get("WANDB_NAME") or output_dir.parent.name,
            "job_type": "gym-observability",
            "dir": str(wandb_dir),
        }
        entity = cfg.get("entity") or os.environ.get("WANDB_ENTITY")
        if entity:
            init_kwargs["entity"] = entity
        wandb.init(**init_kwargs)
        created_run = True
    if wandb.run is None:
        return

    service_name = os.environ.get(
        "NEMO_GYM_OBSERVABILITY_OTEL_SERVICE_NAME",
        "nemo-gym-eval",
    )
    run_id = os.environ.get("NEMO_GYM_OBSERVABILITY_RUN_ID")
    try:
        export_trace_artifacts(output_dir, service_name=service_name, run_id=run_id)
    except Exception:
        pass

    logged_metrics = False
    if cfg.get("log_metrics", True):
        logged_metrics = _log_wandb_metrics(wandb, output_dir, cfg)

    logged_native_outputs = False
    if cfg.get("log_reports", True) or cfg.get("log_tables", True):
        logged_native_outputs = _log_wandb_native_outputs(wandb, output_dir, cfg)

    artifact_logged = False
    if cfg.get("log_artifact", True):
        artifact = wandb.Artifact(
            cfg.get("artifact_name", "gym-observability"),
            type="gym-observability",
        )
        artifact.add_dir(str(output_dir))
        wandb.run.log_artifact(artifact)
        artifact_logged = True

    run_url_attr = getattr(wandb.run, "url", None)
    run_url = run_url_attr if isinstance(run_url_attr, str) else wandb.run.get_url()
    wandb_metadata = {
        "artifact_logged": artifact_logged,
        "artifact_name": cfg.get("artifact_name", "gym-observability"),
        "logged_metrics": logged_metrics,
        "logged_native_outputs": logged_native_outputs,
        "run_id": wandb.run.id,
        "run_name": wandb.run.name,
        "run_url": run_url,
    }
    (output_dir / "wandb.json").write_text(
        json.dumps(wandb_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if created_run:
        wandb.finish()


def _log_wandb_metrics(wandb: Any, output_dir: Path, cfg: dict[str, Any]) -> bool:
    try:
        summary = write_summary(output_dir)
        prefix = str(cfg.get("metric_prefix") or "gym/observability").strip("/")
        last_elapsed_time_s = None
        if cfg.get("log_time_series", True):
            last_elapsed_time_s = _log_wandb_time_series(wandb, output_dir, cfg, prefix=prefix)
        metrics: dict[str, Any] = flatten_summary_metrics(summary, prefix=prefix)
        metrics.update(_wandb_histograms(wandb, output_dir, prefix=prefix))
        if not metrics:
            return False
        if last_elapsed_time_s is not None:
            metrics[f"{prefix}/elapsed_time_s"] = last_elapsed_time_s
        wandb.run.log(metrics)
        summary_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        run_summary = getattr(wandb.run, "summary", None)
        if hasattr(run_summary, "update"):
            run_summary.update(summary_metrics)
        return True
    except Exception:
        return False


def _log_wandb_time_series(
    wandb: Any,
    output_dir: Path,
    cfg: dict[str, Any],
    *,
    prefix: str,
) -> float | None:
    events = load_jsonl(output_dir / "events.jsonl")
    resources = load_jsonl(output_dir / "resource_samples.jsonl")
    rows = _wandb_time_series_rows(events, resources, prefix=prefix)
    if not rows:
        return None
    max_points = int(cfg.get("max_time_series_points") or 0)
    if max_points > 0:
        rows = _sample_time_series_rows(rows, max_points=max_points)
    _define_wandb_time_metrics(wandb, prefix=prefix)
    for _elapsed_time_s, payload in rows:
        wandb.run.log(payload)
    return rows[-1][0]


def _wandb_time_series_rows(
    events: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    *,
    prefix: str,
) -> list[tuple[float, dict[str, Any]]]:
    time_key = f"{prefix}/elapsed_time_s"
    rows: list[tuple[float, dict[str, Any]]] = []
    event_count = 0
    model_call_count = 0
    error_count = 0

    for event in events:
        elapsed_time_s = _row_elapsed_time_s(event)
        if elapsed_time_s is None:
            continue
        attrs = event.get("attributes") or {}
        payload: dict[str, Any] = {time_key: elapsed_time_s}
        if event.get("event_type") == "span_end":
            event_count += 1
            payload[f"{prefix}/events/count"] = 1
            payload[f"{prefix}/events/cumulative_count"] = event_count
            duration_s = attrs.get("duration_s")
            if isinstance(duration_s, (int, float)):
                payload[f"{prefix}/events/duration_s"] = float(duration_s)
                phase = attrs.get("phase")
                if phase:
                    payload[f"{prefix}/events/by_phase/{_safe_metric_component(phase)}/duration_s"] = float(duration_s)
                name = event.get("name")
                if name:
                    payload[f"{prefix}/events/by_name/{_safe_metric_component(name)}/duration_s"] = float(duration_s)
            if attrs.get("status") == "error":
                error_count += 1
                payload[f"{prefix}/errors/count"] = 1
                payload[f"{prefix}/errors/cumulative_count"] = error_count
            if event.get("name") == "llm.request":
                model_call_count += 1
                payload[f"{prefix}/model_calls/count"] = 1
                payload[f"{prefix}/model_calls/cumulative_count"] = model_call_count
                _copy_numeric_metric(payload, attrs, "duration_s", f"{prefix}/model_calls/duration_s")
                _copy_numeric_metric(
                    payload,
                    attrs,
                    "prompt_tokens",
                    f"{prefix}/model_calls/prompt_tokens",
                )
                _copy_numeric_metric(
                    payload,
                    attrs,
                    "completion_tokens",
                    f"{prefix}/model_calls/completion_tokens",
                )
                _copy_numeric_metric(
                    payload,
                    attrs,
                    "total_tokens",
                    f"{prefix}/model_calls/total_tokens",
                )
        elif event.get("name") == "trajectory.complete":
            _copy_numeric_metric(payload, attrs, "reward", f"{prefix}/trajectory/reward")
            _copy_numeric_metric(
                payload,
                attrs,
                "duration_s",
                f"{prefix}/trajectory/duration_s",
            )
            stop_reason = attrs.get("stop_reason")
            if stop_reason:
                payload[f"{prefix}/trajectory/stop_reason/{_safe_metric_component(stop_reason)}"] = 1
        if len(payload) > 1:
            rows.append((elapsed_time_s, payload))

    for sample in resources:
        elapsed_time_s = _row_elapsed_time_s(sample)
        if elapsed_time_s is None:
            continue
        payload = {time_key: elapsed_time_s}
        _copy_numeric_metric(
            payload,
            sample,
            "cpu_utilization",
            f"{prefix}/resources/cpu_utilization",
        )
        _copy_numeric_metric(
            payload,
            sample,
            "memory_usage_bytes",
            f"{prefix}/resources/memory_usage_bytes",
        )
        _copy_numeric_metric(
            payload,
            sample,
            "process_count",
            f"{prefix}/resources/process_count",
        )
        if len(payload) > 1:
            rows.append((elapsed_time_s, payload))

    return sorted(rows, key=lambda item: item[0])


def _copy_numeric_metric(
    payload: dict[str, Any],
    source: dict[str, Any],
    source_key: str,
    metric_key: str,
) -> None:
    value = source.get(source_key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        payload[metric_key] = float(value)


def _row_elapsed_time_s(row: dict[str, Any]) -> float | None:
    value = row.get("elapsed_time_s")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _sample_time_series_rows(
    rows: list[tuple[float, dict[str, Any]]],
    *,
    max_points: int,
) -> list[tuple[float, dict[str, Any]]]:
    if len(rows) <= max_points:
        return rows
    if max_points <= 1:
        return [rows[-1]]
    last_index = len(rows) - 1
    return [rows[round(index * last_index / (max_points - 1))] for index in range(max_points)]


def _define_wandb_time_metrics(wandb: Any, *, prefix: str) -> None:
    time_key = f"{prefix}/elapsed_time_s"
    define_metric = getattr(wandb.run, "define_metric", None) or getattr(
        wandb,
        "define_metric",
        None,
    )
    if define_metric is None:
        return
    try:
        define_metric(time_key)
        define_metric(f"{prefix}/events/*", step_metric=time_key)
        define_metric(f"{prefix}/errors/*", step_metric=time_key)
        define_metric(f"{prefix}/model_calls/*", step_metric=time_key)
        define_metric(f"{prefix}/resources/*", step_metric=time_key)
        define_metric(f"{prefix}/trajectory/*", step_metric=time_key)
    except Exception:
        return


def _wandb_histograms(wandb: Any, output_dir: Path, *, prefix: str) -> dict[str, Any]:
    histogram = getattr(wandb, "Histogram", None)
    if histogram is None:
        return {}
    events = load_jsonl(output_dir / "events.jsonl")
    resources = load_jsonl(output_dir / "resource_samples.jsonl")

    duration_values = []
    llm_duration_values = []
    prompt_tokens = []
    completion_tokens = []
    for event in events:
        attrs = event.get("attributes") or {}
        duration_s = attrs.get("duration_s")
        if isinstance(duration_s, (int, float)):
            duration_values.append(float(duration_s))
            if event.get("name") == "llm.request":
                llm_duration_values.append(float(duration_s))
        prompt_token_count = attrs.get("prompt_tokens")
        if isinstance(prompt_token_count, (int, float)):
            prompt_tokens.append(float(prompt_token_count))
        completion_token_count = attrs.get("completion_tokens")
        if isinstance(completion_token_count, (int, float)):
            completion_tokens.append(float(completion_token_count))

    memory_values = [
        float(sample["memory_usage_bytes"])
        for sample in resources
        if isinstance(sample.get("memory_usage_bytes"), int)
    ]
    cpu_values = [
        float(sample["cpu_utilization"])
        for sample in resources
        if isinstance(sample.get("cpu_utilization"), (int, float))
    ]

    histograms = {}
    _add_histogram(histograms, f"{prefix}/histograms/duration_s", histogram, duration_values)
    _add_histogram(
        histograms,
        f"{prefix}/histograms/llm_request_duration_s",
        histogram,
        llm_duration_values,
    )
    _add_histogram(
        histograms,
        f"{prefix}/histograms/prompt_tokens",
        histogram,
        prompt_tokens,
    )
    _add_histogram(
        histograms,
        f"{prefix}/histograms/completion_tokens",
        histogram,
        completion_tokens,
    )
    _add_histogram(
        histograms,
        f"{prefix}/histograms/memory_usage_bytes",
        histogram,
        memory_values,
    )
    _add_histogram(histograms, f"{prefix}/histograms/cpu_utilization", histogram, cpu_values)
    return histograms


def _add_histogram(
    histograms: dict[str, Any],
    key: str,
    histogram: Any,
    values: list[float],
) -> None:
    if values:
        histograms[key] = histogram(values)


def _log_wandb_native_outputs(wandb: Any, output_dir: Path, cfg: dict[str, Any]) -> bool:
    prefix = str(cfg.get("metric_prefix") or "gym/observability").strip("/")
    payload: dict[str, Any] = {}
    if cfg.get("log_reports", True):
        payload.update(_wandb_report_media(wandb, output_dir, cfg, prefix=prefix))
    if cfg.get("log_tables", True):
        payload.update(_wandb_tables(wandb, output_dir, cfg, prefix=prefix))
    if not payload:
        return False
    try:
        wandb.run.log(payload)
        return True
    except Exception:
        return False


def _wandb_report_media(
    wandb: Any,
    output_dir: Path,
    cfg: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    image_cls = getattr(wandb, "Image", None)
    if image_cls is None:
        return {}
    reports_dir = output_dir / "reports"
    payload: dict[str, Any] = {}
    aggregate_png = reports_dir / "aggregate.png"
    if aggregate_png.exists():
        payload[f"{prefix}/reports/aggregate"] = image_cls(
            str(aggregate_png),
            caption="Aggregate NeMo Gym observability",
        )
    max_media_items = int(cfg.get("max_media_items") or 0)
    trajectory_pngs = sorted(reports_dir.glob("trajectory-*.png"))
    if max_media_items > 0:
        trajectory_pngs = trajectory_pngs[:max_media_items]
    if trajectory_pngs:
        payload[f"{prefix}/reports/trajectories"] = [
            image_cls(str(path), caption=path.stem) for path in trajectory_pngs
        ]
    return payload


def _wandb_tables(
    wandb: Any,
    output_dir: Path,
    cfg: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    table_cls = getattr(wandb, "Table", None)
    if table_cls is None:
        return {}
    max_rows = int(cfg.get("max_table_rows") or 0)
    events = load_jsonl(output_dir / "events.jsonl")
    resources = load_jsonl(output_dir / "resource_samples.jsonl")
    payload: dict[str, Any] = {}
    event_rows = _sample_table_rows(_wandb_event_rows(events), max_rows=max_rows)
    resource_rows = _sample_table_rows(_wandb_resource_rows(resources), max_rows=max_rows)
    model_call_rows = _sample_table_rows(
        _wandb_model_call_rows(events),
        max_rows=max_rows,
    )
    if event_rows:
        payload[f"{prefix}/tables/events"] = table_cls(
            columns=[
                "elapsed_time_s",
                "event_type",
                "name",
                "phase",
                "status",
                "duration_s",
                "command_class",
                "source",
                "error_type",
                "trajectory_id_hash",
                "sandbox_id_hash",
            ],
            data=event_rows,
        )
    if resource_rows:
        payload[f"{prefix}/tables/resources"] = table_cls(
            columns=[
                "elapsed_time_s",
                "cpu_utilization",
                "memory_usage_bytes",
                "process_count",
                "trajectory_id_hash",
                "sandbox_id_hash",
            ],
            data=resource_rows,
        )
    if model_call_rows:
        payload[f"{prefix}/tables/model_calls"] = table_cls(
            columns=[
                "elapsed_time_s",
                "status",
                "duration_s",
                "upstream_api",
                "streaming",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "error_type",
                "trajectory_id_hash",
            ],
            data=model_call_rows,
        )
    return payload


def _wandb_event_rows(events: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for event in events:
        attrs = event.get("attributes") or {}
        rows.append(
            [
                _row_elapsed_time_s(event),
                event.get("event_type"),
                event.get("name"),
                attrs.get("phase"),
                attrs.get("status"),
                attrs.get("duration_s"),
                attrs.get("command_class"),
                attrs.get("source"),
                attrs.get("error_type"),
                _hashed_table_id(attrs.get("trajectory_id")),
                _hashed_table_id(attrs.get("sandbox_id")),
            ]
        )
    return rows


def _wandb_resource_rows(resources: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for sample in resources:
        attrs = sample.get("attributes") or {}
        rows.append(
            [
                _row_elapsed_time_s(sample),
                sample.get("cpu_utilization"),
                sample.get("memory_usage_bytes"),
                sample.get("process_count"),
                _hashed_table_id(attrs.get("trajectory_id")),
                _hashed_table_id(sample.get("sandbox_id") or attrs.get("sandbox_id")),
            ]
        )
    return rows


def _wandb_model_call_rows(events: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for event in events:
        if event.get("name") != "llm.request":
            continue
        attrs = event.get("attributes") or {}
        rows.append(
            [
                _row_elapsed_time_s(event),
                attrs.get("status"),
                attrs.get("duration_s"),
                attrs.get("upstream_api") or attrs.get("upstream_path"),
                attrs.get("streaming"),
                attrs.get("prompt_tokens"),
                attrs.get("completion_tokens"),
                attrs.get("total_tokens"),
                attrs.get("error_type"),
                _hashed_table_id(attrs.get("trajectory_id")),
            ]
        )
    return rows


def _sample_table_rows(rows: list[list[Any]], *, max_rows: int) -> list[list[Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    if max_rows == 1:
        return [rows[-1]]
    last_index = len(rows) - 1
    return [rows[round(index * last_index / (max_rows - 1))] for index in range(max_rows)]


def _hashed_table_id(value: Any) -> str | None:
    if value is None:
        return None
    return stable_hash(str(value), length=12)


def build_recorder_from_config(
    config: dict[str, Any] | None,
    *,
    run_id: str | None = None,
) -> SandboxEventRecorder | None:
    """Build a recorder from a Hydra observability config dict.

    Returns None when observability is disabled or ``output_dir`` is not set.
    """
    if not isinstance(config, dict):
        return None
    output_dir = config.get("output_dir") or config.get("dir")
    if not output_dir:
        return None
    enabled = config.get("enabled", True)
    if not enabled:
        return None
    return SandboxEventRecorder(
        output_dir=Path(output_dir),
        resource_sample_interval_s=float(config.get("resource_sample_interval_s", 10)),
        max_rendered_trajectories=int(config.get("max_rendered_trajectories", 40)),
        artifacts=config.get("artifacts")
        or {
            "enabled": True,
            "render_html": True,
            "render_png": True,
            "export_otlp_json": True,
        },
        otel=config.get("otel")
        or {
            "enabled": False,
            "endpoint": None,
            "service_name": "nemo-gym-eval",
        },
        wandb=config.get("wandb")
        or {
            "enabled": None,
            "artifact_name": "gym-observability",
            "log_metrics": True,
            "log_time_series": True,
            "max_time_series_points": 10000,
            "log_reports": True,
            "max_media_items": 20,
            "log_tables": True,
            "max_table_rows": 10000,
            "metric_prefix": "gym/observability",
        },
        process_trace=config.get("process_trace")
        or {
            "enabled": False,
            "sample_interval_s": 1.0,
            "max_processes_per_sample": 128,
            "include_cmdline": False,
        },
        privacy=config.get("privacy") or {"include_command_text": False},
        run_id=run_id,
    )


def build_recorder_from_env() -> SandboxEventRecorder | None:
    """Build a recorder from NEMO_GYM_OBSERVABILITY_* environment variables."""
    output_dir = os.environ.get("NEMO_GYM_OBSERVABILITY_DIR")
    if not output_dir:
        return None
    return SandboxEventRecorder(
        output_dir=Path(output_dir),
        resource_sample_interval_s=float(os.environ.get("NEMO_GYM_OBSERVABILITY_RESOURCE_INTERVAL_S", "10")),
        max_rendered_trajectories=int(os.environ.get("NEMO_GYM_OBSERVABILITY_MAX_RENDERED", "40")),
        artifacts={
            "enabled": os.environ.get("NEMO_GYM_OBSERVABILITY_ARTIFACTS", "1") != "0",
            "render_html": os.environ.get("NEMO_GYM_OBSERVABILITY_RENDER_HTML", "1") != "0",
            "render_png": os.environ.get("NEMO_GYM_OBSERVABILITY_RENDER_PNG", "1") != "0",
            "export_otlp_json": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_EXPORT_OTLP_JSON",
                "1",
            )
            != "0",
        },
        otel={
            "enabled": bool(os.environ.get("NEMO_GYM_OBSERVABILITY_OTEL_ENDPOINT")),
            "service_name": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_OTEL_SERVICE_NAME",
                "nemo-gym-eval",
            ),
            "endpoint": os.environ.get("NEMO_GYM_OBSERVABILITY_OTEL_ENDPOINT"),
            "export_logs": os.environ.get("NEMO_GYM_OBSERVABILITY_OTEL_LOGS", "0") == "1",
        },
        wandb={
            "enabled": _env_wandb_enabled(os.environ.get("NEMO_GYM_OBSERVABILITY_WANDB", "auto")),
            "artifact_name": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_WANDB_ARTIFACT",
                "gym-observability",
            ),
            "project": os.environ.get("NEMO_GYM_OBSERVABILITY_WANDB_PROJECT"),
            "entity": os.environ.get("NEMO_GYM_OBSERVABILITY_WANDB_ENTITY"),
            "run_name": os.environ.get("NEMO_GYM_OBSERVABILITY_WANDB_RUN_NAME"),
            "log_artifact": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_WANDB_LOG_ARTIFACT",
                "1",
            )
            != "0",
            "log_metrics": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_WANDB_LOG_METRICS",
                "1",
            )
            != "0",
            "log_time_series": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_WANDB_LOG_TIME_SERIES",
                "1",
            )
            != "0",
            "max_time_series_points": int(
                os.environ.get(
                    "NEMO_GYM_OBSERVABILITY_WANDB_MAX_TIME_SERIES_POINTS",
                    "10000",
                )
            ),
            "log_reports": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_WANDB_LOG_REPORTS",
                "1",
            )
            != "0",
            "max_media_items": int(
                os.environ.get(
                    "NEMO_GYM_OBSERVABILITY_WANDB_MAX_MEDIA_ITEMS",
                    "20",
                )
            ),
            "log_tables": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_WANDB_LOG_TABLES",
                "1",
            )
            != "0",
            "max_table_rows": int(
                os.environ.get(
                    "NEMO_GYM_OBSERVABILITY_WANDB_MAX_TABLE_ROWS",
                    "10000",
                )
            ),
            "metric_prefix": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_WANDB_METRIC_PREFIX",
                "gym/observability",
            ),
        },
        process_trace={
            "enabled": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_PROCESS_TRACE",
                "0",
            )
            == "1",
            "sample_interval_s": float(
                os.environ.get(
                    "NEMO_GYM_OBSERVABILITY_PROCESS_TRACE_INTERVAL_S",
                    "1.0",
                )
            ),
            "max_processes_per_sample": int(
                os.environ.get(
                    "NEMO_GYM_OBSERVABILITY_PROCESS_TRACE_MAX_PROCESSES",
                    "128",
                )
            ),
            "include_cmdline": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_PROCESS_TRACE_INCLUDE_CMDLINE",
                "0",
            )
            == "1",
        },
        privacy={
            "include_command_text": os.environ.get(
                "NEMO_GYM_OBSERVABILITY_INCLUDE_COMMAND_TEXT",
                "0",
            )
            == "1"
        },
        run_id=os.environ.get("NEMO_GYM_OBSERVABILITY_RUN_ID"),
    )


def _env_wandb_enabled(value: str) -> bool | None:
    if value.lower() in {"auto", ""}:
        return None
    return value not in {"0", "false", "False"}


def ensure_env_recorder() -> SandboxEventRecorder | None:
    """Return the process-wide env-configured recorder, creating it once."""
    global G_ENV_RECORDER
    if G_ENV_RECORDER is not None:
        return G_ENV_RECORDER
    with G_ENV_RECORDER_LOCK:
        if G_ENV_RECORDER is None:
            G_ENV_RECORDER = build_recorder_from_env()
            if G_ENV_RECORDER is not None:
                atexit.register(G_ENV_RECORDER.finalize)
    return G_ENV_RECORDER


def current_recorder() -> SandboxEventRecorder | None:
    """Return the active context recorder, if any."""
    return G_CURRENT_RECORDER.get()


def _active_recorder() -> SandboxEventRecorder | None:
    return current_recorder() or ensure_env_recorder()


def set_current_recorder(recorder: SandboxEventRecorder) -> Token[SandboxEventRecorder | None]:
    """Set the active recorder for the current context."""
    return G_CURRENT_RECORDER.set(recorder)


def reset_current_recorder(token: Token[SandboxEventRecorder | None]) -> None:
    """Reset the active recorder token."""
    G_CURRENT_RECORDER.reset(token)


@contextmanager
def use_recorder(recorder: SandboxEventRecorder | None) -> Iterator[None]:
    """Temporarily set the current recorder."""
    if recorder is None:
        yield
        return
    token = set_current_recorder(recorder)
    try:
        yield
    finally:
        reset_current_recorder(token)


def push_event_context(attributes: dict[str, Any]) -> Token[dict[str, Any]]:
    """Merge event context attributes for the current task."""
    return G_EVENT_CONTEXT.set({**G_EVENT_CONTEXT.get(), **attributes})


def reset_event_context(token: Token[dict[str, Any]]) -> None:
    """Reset event context attributes."""
    G_EVENT_CONTEXT.reset(token)


@contextmanager
def event_context(**attributes: Any) -> Iterator[None]:
    """Temporarily add ambient event context attributes."""
    token = push_event_context(attributes)
    try:
        yield
    finally:
        reset_event_context(token)


@contextmanager
def suppress_observability_events() -> Iterator[None]:
    """Suppress nested events (used by resource sampler to avoid self-recording)."""
    token = G_SUPPRESS_EVENTS.set(G_SUPPRESS_EVENTS.get() + 1)
    try:
        yield
    finally:
        G_SUPPRESS_EVENTS.reset(token)


def observability_suppressed() -> bool:
    """Return whether event recording is suppressed in this context."""
    return G_SUPPRESS_EVENTS.get() > 0


def record_event(
    event_type: str,
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record one event on the current recorder."""
    recorder = _active_recorder()
    if recorder is not None and not observability_suppressed():
        recorder.record_event(event_type, name, attributes=attributes)


@asynccontextmanager
async def observability_span(
    name: str,
    *,
    phase: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record an async span on the current recorder (JSONL + OTel trace)."""
    recorder = _active_recorder()
    if recorder is None or observability_suppressed():
        yield
        return
    async with recorder.span(name, phase=phase, attributes=attributes):
        yield


@contextmanager
def observability_sync_span(
    name: str,
    *,
    phase: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record a sync span on the current recorder (JSONL + OTel trace)."""
    recorder = _active_recorder()
    if recorder is None or observability_suppressed():
        yield
        return
    with recorder.sync_span(name, phase=phase, attributes=attributes):
        yield
