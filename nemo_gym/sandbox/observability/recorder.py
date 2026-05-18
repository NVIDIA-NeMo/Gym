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

"""Context-scoped recorder for sandbox eval observability."""

from __future__ import annotations

import atexit
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from nemo_gym.sandbox.observability.traces import (
    SCOPE_NAME,
    SCOPE_VERSION,
    JsonSpanExporter,
    export_trace_artifacts,
)


G_CURRENT_RECORDER: ContextVar[SandboxRecorder | None] = ContextVar(
    "nemo_gym_sandbox_observability_recorder",
    default=None,
)
G_EVENT_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "nemo_gym_sandbox_observability_context",
    default={},
)
G_ENV_RECORDER: SandboxRecorder | None = None
G_ENV_RECORDER_LOCK = threading.Lock()


class SandboxRecorder:
    """Context-scoped sandbox observability recorder backed by OpenTelemetry."""

    def __init__(
        self,
        *,
        output_dir: Path,
        otel: dict[str, Any] | None = None,
        run_id: str | None = None,
        export_traces: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.otel = dict(otel or {})
        self.run_id = run_id
        self.export_traces = export_traces
        self.attribute_aliases = _string_map(self.otel.get("attribute_aliases"))
        self.metric_attribute_keys = tuple(str(key) for key in self.otel.get("metric_attribute_keys") or ())
        self.resource_attributes = safe_attributes(self.otel.get("resource_attributes") or {})
        self._closed = False
        self._service_name = str(self.otel.get("service_name") or "") or None
        self._span_exporter = JsonSpanExporter()
        self._tracer_provider = TracerProvider(resource=self._resource())
        self._tracer_provider.add_span_processor(SimpleSpanProcessor(self._span_exporter))
        self._meter_provider = None
        self._duration_histogram = None
        self._phase_duration_histograms = {}
        self._counter = None
        self._configure_live_exporters()
        self._configure_metrics()
        self._trajectory_spans: dict[str, Span] = {}
        self._run_span = self._start_span(
            "sandbox.run",
            attributes=safe_attributes({"run_id": run_id}),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_event("lifecycle", "run.start", attributes={"run_id": run_id})

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
    ) -> None:
        """Record one OpenTelemetry event on the active span."""
        del trace_id, span_id, parent_span_id, monotonic_s
        attrs = safe_attributes({**G_EVENT_CONTEXT.get(), **(attributes or {})})
        self._record_otel_event(event_type, name, attrs, timestamp_unix_s=timestamp_unix_s)
        self._record_event_metrics(event_type=event_type, name=name, attrs=attrs)

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        phase: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Record an async operation as an OpenTelemetry span."""
        with self._span_context(name, phase=phase, attributes=attributes):
            yield

    @contextmanager
    def sync_span(
        self,
        name: str,
        *,
        phase: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Record a synchronous operation as an OpenTelemetry span."""
        with self._span_context(name, phase=phase, attributes=attributes):
            yield

    def finalize(self) -> None:
        """Flush local trace artifacts and OTel exporters."""
        if self._closed:
            return
        self._closed = True
        self.record_event("lifecycle", "run.end", attributes={"run_id": self.run_id})
        self._end_open_spans()
        if self.export_traces:
            try:
                self._export_trace_artifacts()
            except Exception as e:
                self.record_event(
                    "error",
                    "observability.trace_export_error",
                    attributes={"error_type": type(e).__name__, "error": str(e)},
                )
        self._shutdown_otel()

    @contextmanager
    def _span_context(
        self,
        name: str,
        *,
        phase: str | None,
        attributes: dict[str, Any] | None,
    ) -> Iterator[None]:
        start_monotonic = time.monotonic()
        span_attrs = safe_attributes({**G_EVENT_CONTEXT.get(), "phase": phase, **(attributes or {})})
        with self._start_as_current_span(
            name,
            attributes=self._span_attributes(name, span_attrs),
            context=self._parent_context(span_attrs),
            kind=_span_kind(name),
        ) as span:
            try:
                yield
            except Exception as e:
                duration_s = time.monotonic() - start_monotonic
                span.record_exception(e)
                span.set_attribute("duration_s", duration_s)
                span.set_attribute("status", "error")
                span.set_status(Status(StatusCode.ERROR, type(e).__name__))
                self._record_span_metrics(
                    name=name,
                    attrs={**span_attrs, "status": "error", "duration_s": duration_s},
                )
                raise
            else:
                duration_s = time.monotonic() - start_monotonic
                span.set_attribute("duration_s", duration_s)
                span.set_attribute("status", "ok")
                span.set_status(Status(StatusCode.OK))
                self._record_span_metrics(
                    name=name,
                    attrs={**span_attrs, "status": "ok", "duration_s": duration_s},
                )

    def _record_otel_event(
        self,
        event_type: str,
        name: str,
        attrs: dict[str, Any],
        *,
        timestamp_unix_s: float | None = None,
    ) -> None:
        event_attrs = self._span_attributes(name, {"event.type": event_type, **attrs})
        timestamp_ns = _timestamp_ns(timestamp_unix_s)
        current_span = trace.get_current_span()
        if current_span is not None and current_span.is_recording():
            current_span.add_event(name, event_attrs, timestamp=timestamp_ns)
            self._maybe_close_trajectory_span(name, attrs)
            return

        parent_span = self._event_parent_span(attrs)
        if parent_span is not None and parent_span.is_recording():
            parent_span.add_event(name, event_attrs, timestamp=timestamp_ns)
            self._maybe_close_trajectory_span(name, attrs)
            return

        with self._start_as_current_span(
            name,
            attributes=event_attrs,
            context=self._parent_context(attrs),
            kind=_span_kind(name),
        ) as span:
            span.add_event(name, event_attrs, timestamp=timestamp_ns)
            if event_type == "error":
                span.set_status(Status(StatusCode.ERROR))
        self._maybe_close_trajectory_span(name, attrs)

    def _event_parent_span(self, attrs: dict[str, Any]) -> Span | None:
        trajectory_id = _trajectory_id(attrs)
        if trajectory_id:
            return self._trajectory_span(trajectory_id, attrs)
        return self._run_span if self._run_span.is_recording() else None

    def _parent_context(self, attrs: dict[str, Any]) -> Any:
        parent_span = self._event_parent_span(attrs)
        return trace.set_span_in_context(parent_span) if parent_span is not None else None

    def _trajectory_span(self, trajectory_id: str, attrs: dict[str, Any]) -> Span:
        span = self._trajectory_spans.get(trajectory_id)
        if span is not None and span.is_recording():
            _set_span_attributes(span, self._trajectory_root_attributes(trajectory_id, attrs))
            return span
        span = self._start_span(
            "trajectory",
            attributes=self._trajectory_root_attributes(trajectory_id, attrs),
            context=trace.set_span_in_context(self._run_span),
        )
        self._trajectory_spans[trajectory_id] = span
        return span

    def _trajectory_root_attributes(self, trajectory_id: str, attrs: dict[str, Any]) -> dict[str, Any]:
        root_attrs = {
            "event.type": "synthetic_root",
            "phase": "trajectory",
            "trajectory_id": trajectory_id,
        }
        for key in (
            "reward",
            "stop_reason",
            "duration_s",
            "loss_multiplier",
            "attempt_idx",
            "harness",
            "dataset_alias",
        ):
            if attrs.get(key) is not None:
                root_attrs[key] = attrs[key]
        return safe_attributes(root_attrs)

    def _span_attributes(self, name: str, attrs: dict[str, Any]) -> dict[str, Any]:
        span_attrs = safe_attributes({**attrs})
        if self.run_id:
            span_attrs.setdefault("run_id", self.run_id)
        for source, target in self.attribute_aliases.items():
            if source in span_attrs and span_attrs[source] is not None:
                span_attrs.setdefault(target, span_attrs[source])
        return span_attrs

    def _maybe_close_trajectory_span(self, name: str, attrs: dict[str, Any]) -> None:
        if name not in {"trajectory.complete", "trajectory.masked"}:
            return
        trajectory_id = _trajectory_id(attrs)
        if trajectory_id is None:
            return
        span = self._trajectory_spans.pop(trajectory_id, None)
        if span is None or not span.is_recording():
            return
        _set_span_attributes(span, self._trajectory_root_attributes(trajectory_id, attrs))
        span.set_status(Status(StatusCode.ERROR if attrs.get("stop_reason") == "error" else StatusCode.OK))
        span.end()

    def _end_open_spans(self) -> None:
        for trajectory_id, span in list(self._trajectory_spans.items()):
            if span.is_recording():
                span.set_attribute("stop_reason", "observability_finalize")
                span.end()
            self._trajectory_spans.pop(trajectory_id, None)
        if self._run_span.is_recording():
            self._run_span.set_status(Status(StatusCode.OK))
            self._run_span.end()

    def _resource(self) -> Resource:
        attributes = dict(self.resource_attributes)
        if self._service_name:
            attributes["service.name"] = self._service_name
        return Resource.create(attributes)

    def _configure_live_exporters(self) -> None:
        if not self.otel.get("enabled"):
            return
        endpoint = _otel_trace_endpoint(self.otel)
        if not endpoint:
            return
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            return
        self._tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    def _configure_metrics(self) -> None:
        endpoint = _otel_metric_endpoint(self.otel)
        readers = []
        if self.otel.get("enabled") and endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            except ImportError:
                readers = []
            else:
                readers = [PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))]
        self._meter_provider = MeterProvider(resource=self._resource(), metric_readers=readers)
        meter = self._meter_provider.get_meter(SCOPE_NAME, SCOPE_VERSION)
        self._duration_histogram = meter.create_histogram(
            "nemo_gym.sandbox.operation.duration",
            unit="s",
            description="Sandbox operation duration.",
        )
        self._phase_duration_histograms = {
            phase: meter.create_histogram(
                f"nemo_gym.sandbox.{phase}.duration",
                unit="s",
                description=f"Sandbox {phase} duration.",
            )
            for phase in ("startup", "setup", "execution", "llm")
        }
        self._counter = meter.create_counter(
            "nemo_gym.sandbox.events",
            description="Sandbox event counts.",
        )

    def _start_span(self, name: str, *, attributes: dict[str, Any], context: Any = None) -> Span:
        return self._tracer_provider.get_tracer(SCOPE_NAME, SCOPE_VERSION).start_span(
            name,
            context=context,
            attributes=attributes,
            kind=_span_kind(name),
        )

    def _start_as_current_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any],
        context: Any = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Any:
        return self._tracer_provider.get_tracer(SCOPE_NAME, SCOPE_VERSION).start_as_current_span(
            name,
            context=context,
            attributes=attributes,
            kind=kind,
        )

    def _record_event_metrics(self, *, event_type: str, name: str, attrs: dict[str, Any]) -> None:
        otel_attrs = self._metric_attrs(attrs)
        otel_attrs["event_name"] = name
        otel_attrs["event_type"] = event_type
        if self._counter is not None:
            self._counter.add(1, otel_attrs)

    def _record_span_metrics(self, *, name: str, attrs: dict[str, Any]) -> None:
        otel_attrs = self._metric_attrs(attrs)
        otel_attrs["span_name"] = name
        if self._counter is not None:
            self._counter.add(1, otel_attrs)
        duration_s = attrs.get("duration_s")
        if self._duration_histogram is not None and isinstance(duration_s, (int, float)):
            self._duration_histogram.record(float(duration_s), otel_attrs)
            phase_histogram = self._phase_duration_histograms.get(str(attrs.get("phase") or ""))
            if phase_histogram is not None:
                phase_histogram.record(float(duration_s), otel_attrs)

    def _metric_attrs(self, attrs: dict[str, Any]) -> dict[str, str]:
        return {key: str(attrs[key]) for key in self.metric_attribute_keys if key in attrs and attrs[key] is not None}

    def _export_trace_artifacts(self) -> dict[str, str]:
        self._tracer_provider.force_flush()
        return export_trace_artifacts(
            self.output_dir,
            spans=self._span_exporter.finished_spans(),
        )

    def _shutdown_otel(self) -> None:
        self._tracer_provider.shutdown()
        if self._meter_provider is not None:
            self._meter_provider.shutdown()


def _trajectory_id(attrs: dict[str, Any]) -> str | None:
    value = attrs.get("trajectory_id") or attrs.get("trial_name")
    return str(value) if value else None


def _set_span_attributes(span: Span, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        span.set_attribute(key, value)


def _span_kind(name: str) -> SpanKind:
    return SpanKind.CLIENT if name == "llm.request" else SpanKind.INTERNAL


def _timestamp_ns(timestamp_unix_s: float | None) -> int | None:
    return int(timestamp_unix_s * 1_000_000_000) if timestamp_unix_s is not None else None


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(source): str(target) for source, target in value.items()}


def safe_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Return OpenTelemetry-friendly attributes."""
    if not attributes:
        return {}
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = [
                item if isinstance(item, (str, int, float, bool)) or item is None else str(item) for item in value
            ]
        else:
            safe[key] = str(value)
    return safe


def _otel_trace_endpoint(cfg: dict[str, Any]) -> str | None:
    return _otel_signal_endpoint(cfg.get("traces_endpoint") or cfg.get("endpoint"), signal="traces")


def _otel_metric_endpoint(cfg: dict[str, Any]) -> str | None:
    return _otel_signal_endpoint(cfg.get("metrics_endpoint") or cfg.get("endpoint"), signal="metrics")


def _otel_signal_endpoint(endpoint: Any, *, signal: str) -> str | None:
    if not endpoint:
        return None
    endpoint_str = str(endpoint).rstrip("/")
    if endpoint_str.endswith(f"/v1/{signal}"):
        return endpoint_str
    if endpoint_str.endswith("/v1/traces") or endpoint_str.endswith("/v1/metrics"):
        return endpoint_str.rsplit("/v1/", 1)[0] + f"/v1/{signal}"
    return f"{endpoint_str}/v1/{signal}"


def _otel_config_from_env() -> dict[str, Any]:
    traces_endpoint = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_TRACES_ENDPOINT")
    metrics_endpoint = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_METRICS_ENDPOINT")
    endpoint = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_ENDPOINT")
    return {
        "enabled": bool(endpoint or traces_endpoint or metrics_endpoint),
        "service_name": os.environ.get(
            "NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_SERVICE_NAME",
            "",
        ),
        "endpoint": endpoint,
        "traces_endpoint": traces_endpoint,
        "metrics_endpoint": metrics_endpoint,
    }


def build_recorder_from_config(
    config: dict[str, Any] | None,
    *,
    run_id: str | None = None,
) -> SandboxRecorder | None:
    """Build a recorder from a sandbox observability config."""
    if not isinstance(config, dict) or not config.get("enabled", False):
        return None
    output_dir = config.get("output_dir")
    if not output_dir:
        raise ValueError("env.sandbox.observability.output_dir is required when enabled")
    return SandboxRecorder(
        output_dir=Path(output_dir),
        otel=dict(config.get("otel") or {}),
        run_id=run_id,
        export_traces=bool(config.get("export_traces", True)),
    )


def build_recorder_from_env() -> SandboxRecorder | None:
    """Build a recorder from eval-job environment variables."""
    output_dir = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_DIR")
    if not output_dir:
        return None
    return SandboxRecorder(
        output_dir=Path(output_dir),
        otel=_otel_config_from_env(),
        run_id=os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_RUN_ID"),
        export_traces=os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_EXPORT_TRACES", "1") != "0",
    )


def ensure_env_recorder() -> SandboxRecorder | None:
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


def current_recorder() -> SandboxRecorder | None:
    """Return the active context recorder, if any."""
    return G_CURRENT_RECORDER.get()


def _active_recorder() -> SandboxRecorder | None:
    return current_recorder() or ensure_env_recorder()


def set_current_recorder(recorder: SandboxRecorder) -> Token[SandboxRecorder | None]:
    """Set the active recorder for the current context."""
    return G_CURRENT_RECORDER.set(recorder)


def reset_current_recorder(token: Token[SandboxRecorder | None]) -> None:
    """Reset the active recorder token."""
    G_CURRENT_RECORDER.reset(token)


@contextmanager
def use_recorder(recorder: SandboxRecorder | None) -> Iterator[None]:
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
    """Temporarily add event context attributes."""
    token = push_event_context(attributes)
    try:
        yield
    finally:
        reset_event_context(token)


def record_event(
    event_type: str,
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record one event on the current recorder."""
    recorder = _active_recorder()
    if recorder is not None:
        recorder.record_event(event_type, name, attributes=attributes)


@asynccontextmanager
async def observability_span(
    name: str,
    *,
    phase: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record an async span on the current recorder."""
    recorder = _active_recorder()
    if recorder is None:
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
    """Record a sync span on the current recorder."""
    recorder = _active_recorder()
    if recorder is None:
        yield
        return
    with recorder.sync_span(name, phase=phase, attributes=attributes):
        yield
