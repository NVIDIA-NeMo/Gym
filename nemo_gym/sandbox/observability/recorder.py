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
import json
import os
import re
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
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
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
        output_dir: Path | None = None,
        otel: dict[str, Any] | None = None,
        run_id: str | None = None,
        run_span_name: str | None = None,
        export_traces: bool | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.otel = dict(otel or {})
        self.run_id = run_id
        self.run_span_name = _eval_span_name(
            run_span_name or self.otel.get("run_span_name") or self.otel.get("job_name") or run_id or "sandbox.run"
        )
        self._configured_trace_exporters = _configured_exporters(self.otel, "traces")
        self._configured_metric_exporters = _configured_exporters(self.otel, "metrics")
        self.export_traces = _local_trace_export_enabled(
            output_dir=output_dir,
            export_traces=export_traces,
            trace_exporters=self._configured_trace_exporters,
        )
        if self.export_traces and output_dir is None:
            raise ValueError("sandbox observability output_dir is required for local trace export")
        self.attribute_aliases = _string_map(self.otel.get("attribute_aliases"))
        self.command_titles = _command_title_config(self.otel.get("command_titles") or self.otel.get("command_title"))
        self.metric_attribute_keys = tuple(str(key) for key in self.otel.get("metric_attribute_keys") or ())
        self.resource_attributes = safe_attributes(self.otel.get("resource_attributes") or {})
        self.local_service_name_strategy = str(self.otel.get("local_service_name_strategy") or "span_section")
        self._closed = False
        self._service_name = str(self.otel.get("service_name") or "") or None
        self._local_span_exporter = JsonSpanExporter() if self.export_traces else None
        self._tracer_provider = TracerProvider(resource=self._resource())
        if self._local_span_exporter is not None:
            self._tracer_provider.add_span_processor(SimpleSpanProcessor(self._local_span_exporter))
        self._meter_provider = None
        self._duration_histogram = None
        self._phase_duration_histograms = {}
        self._counter = None
        self._configure_live_exporters()
        self._configure_metrics()
        self._trajectory_spans: dict[tuple[str, str], Span] = {}
        self._run_span = self._start_span(
            self.run_span_name,
            attributes=safe_attributes(
                {
                    "run_id": run_id,
                    "span.role": "eval.run",
                    "span.section": "eval",
                }
            ),
        )
        if self.output_dir is not None:
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
        record_exception_stacktrace = _as_bool(span_attrs.pop("_record_exception_stacktrace", True))
        operation_name = _span_name(name)
        display_name = self._operation_span_name(operation_name, span_attrs)
        with self._start_as_current_span(
            display_name,
            attributes=self._span_attributes(operation_name, span_attrs, display_name=display_name),
            context=self._parent_context(span_attrs),
            kind=_span_kind(operation_name),
        ) as span:
            try:
                yield
            except Exception as e:
                duration_s = time.monotonic() - start_monotonic
                if record_exception_stacktrace:
                    span.record_exception(e)
                else:
                    span.add_event(
                        "exception",
                        {
                            "exception.type": type(e).__name__,
                            "exception.message": str(e),
                        },
                    )
                span.set_attribute("duration_s", duration_s)
                span.set_attribute("status", "error")
                span.set_attribute("error_type", type(e).__name__)
                span.set_status(Status(StatusCode.ERROR, type(e).__name__))
                self._record_span_metrics(
                    name=operation_name,
                    attrs={**span_attrs, "status": "error", "duration_s": duration_s},
                )
                raise
            else:
                duration_s = time.monotonic() - start_monotonic
                span.set_attribute("duration_s", duration_s)
                span.set_attribute("status", "ok")
                span.set_status(Status(StatusCode.OK))
                self._record_span_metrics(
                    name=operation_name,
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
            return self._trajectory_span(trajectory_id, attrs, section=_span_section("trajectory", attrs))
        return self._run_span if self._run_span.is_recording() else None

    def _parent_context(self, attrs: dict[str, Any]) -> Any:
        parent_span = self._event_parent_span(attrs)
        return trace.set_span_in_context(parent_span) if parent_span is not None else None

    def _trajectory_span(self, trajectory_id: str, attrs: dict[str, Any], *, section: str) -> Span:
        span_key = (trajectory_id, section)
        span = self._trajectory_spans.get(span_key)
        if span is not None and span.is_recording():
            _set_span_attributes(span, self._trajectory_root_attributes(trajectory_id, attrs, section=section))
            return span
        span = self._start_span(
            _section_span_name(section, trajectory_id),
            attributes=self._trajectory_root_attributes(trajectory_id, attrs, section=section),
            context=self._trajectory_parent_context(trajectory_id, attrs, section=section),
        )
        self._trajectory_spans[span_key] = span
        return span

    def _trajectory_parent_context(self, trajectory_id: str, attrs: dict[str, Any], *, section: str) -> Any:
        if section == "rollout":
            return trace.set_span_in_context(self._run_span)
        rollout_span = self._trajectory_span(trajectory_id, attrs, section="rollout")
        return trace.set_span_in_context(rollout_span)

    def _trajectory_root_attributes(
        self, trajectory_id: str, attrs: dict[str, Any], *, section: str
    ) -> dict[str, Any]:
        root_attrs = {
            "event.type": "synthetic_root",
            "phase": section,
            "span.role": f"{section}.trajectory",
            "span.section": section,
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

    def _span_attributes(self, name: str, attrs: dict[str, Any], *, display_name: str | None = None) -> dict[str, Any]:
        span_attrs = safe_attributes({**attrs})
        span_attrs.setdefault("operation.name", name)
        if display_name is not None and display_name != name:
            span_attrs.setdefault("span.display_name", display_name)
        span_attrs.setdefault("span.section", _span_section(name, span_attrs))
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
        for span_key, span in list(self._trajectory_spans.items()):
            if span_key[0] != trajectory_id:
                continue
            self._trajectory_spans.pop(span_key, None)
            if span is None or not span.is_recording():
                continue
            _set_span_attributes(span, self._trajectory_root_attributes(trajectory_id, attrs, section=span_key[1]))
            span.set_status(Status(StatusCode.ERROR if attrs.get("stop_reason") == "error" else StatusCode.OK))
            span.end()

    def _end_open_spans(self) -> None:
        for span_key, span in list(self._trajectory_spans.items()):
            if span.is_recording():
                span.set_attribute("stop_reason", "observability_finalize")
                span.end()
            self._trajectory_spans.pop(span_key, None)
        if self._run_span.is_recording():
            self._run_span.set_status(Status(StatusCode.OK))
            self._run_span.end()

    def _operation_span_name(self, name: str, attrs: dict[str, Any]) -> str:
        configured_name = attrs.get("span.name")
        if configured_name:
            return _span_name(configured_name)
        if name == "trajectory.tool":
            return f"exec: {_command_title(attrs.get('command'), self.command_titles)}"
        if name == "sandbox.start":
            return _span_with_detail("sandbox.create", attrs.get("image"))
        if name == "sandbox.start_batch":
            count = attrs.get("count")
            detail = f"{count} sandbox{'es' if count != 1 else ''}" if count is not None else None
            return _span_with_detail("sandbox.create_batch", detail)
        if name == "sandbox.cleanup":
            return _span_with_detail("sandbox.cleanup", attrs.get("sandbox_id"))
        return _span_name(name)

    def _resource(self) -> Resource:
        attributes = dict(self.resource_attributes)
        if self._service_name:
            attributes["service.name"] = self._service_name
        return Resource.create(attributes)

    def _configure_live_exporters(self) -> None:
        for exporter_name in _live_trace_exporters(self.otel, self._configured_trace_exporters):
            exporter = _build_trace_exporter(exporter_name, self.otel)
            if exporter is not None:
                self._tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

    def _configure_metrics(self) -> None:
        readers = []
        for exporter_name in _live_metric_exporters(self.otel, self._configured_metric_exporters):
            exporter = _build_metric_exporter(exporter_name, self.otel)
            if exporter is not None:
                readers.append(PeriodicExportingMetricReader(exporter))
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
        if self.output_dir is None or self._local_span_exporter is None:
            return {}
        self._tracer_provider.force_flush()
        return export_trace_artifacts(
            self.output_dir,
            spans=self._local_span_exporter.finished_spans(),
            service_name_strategy=self.local_service_name_strategy,
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


def _span_name(value: Any) -> str:
    span_name = str(value or "").strip()
    return span_name or "sandbox.run"


def _eval_span_name(value: Any) -> str:
    name = _span_name(value)
    return name if name.startswith("eval: ") else f"eval: {name}"


def _section_span_name(section: str, trajectory_id: Any) -> str:
    return f"{section}: {_span_name(trajectory_id)}"


def _span_with_detail(name: str, detail: Any, *, max_length: int = 120) -> str:
    text = _compact_text(detail)
    if not text:
        return name
    title = f"{name}: {text}"
    return title if len(title) <= max_length else f"{title[: max_length - 1].rstrip()}..."


def _command_title(command: Any, config: dict[str, Any]) -> str:
    max_length = int(config.get("max_length") or 140)
    text = _strip_command_prefixes(_compact_text(command), config.get("strip_prefixes") or ())
    if not text:
        return "<empty>"
    configured_title = _configured_command_title(text, config.get("rules") or ())
    if configured_title:
        return _truncate_span_title(configured_title, max_length=max_length)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    title = " ".join(first_line.split())
    return _truncate_span_title(title, max_length=max_length)


def _strip_command_prefixes(command: str, prefixes: Any) -> str:
    text = command
    for prefix in prefixes:
        prefix = str(prefix)
        if text.startswith(prefix):
            return text[len(prefix) :].lstrip()
    return text


def _configured_command_title(command: str, rules: Any) -> str | None:
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        line = _matching_rule_line(command, rule)
        if line is not None:
            return _format_rule_title(rule, command=command, line=line, match=line)
        match = _matching_rule_text(command, rule)
        if match is not None:
            return _format_rule_title(rule, command=command, line="", match=match)
    return None


def _matching_rule_line(command: str, rule: dict[str, Any]) -> str | None:
    line_prefixes = _string_tuple(rule.get("line_starts_with"))
    line_regex = str(rule.get("line_regex") or "")
    if not line_prefixes and not line_regex:
        return None
    lines = [line.strip() for line in command.splitlines() if line.strip()]
    if str(rule.get("search") or "first").lower() == "last":
        lines = list(reversed(lines))
    regex = re.compile(line_regex) if line_regex else None
    for line in lines:
        if line_prefixes and line.startswith(line_prefixes):
            return " ".join(line.split())
        if regex is not None and regex.search(line):
            return " ".join(line.split())
    return None


def _matching_rule_text(command: str, rule: dict[str, Any]) -> str | None:
    contains = _string_tuple(rule.get("contains") or rule.get("all_contains"))
    if contains and not all(part in command for part in contains):
        return None
    starts_with = _string_tuple(rule.get("starts_with"))
    if starts_with and not command.startswith(starts_with):
        return None
    regex = str(rule.get("regex") or "")
    if regex:
        match = re.search(regex, command)
        if match is None:
            return None
        return match.group(0)
    if contains or starts_with:
        return command
    return None


def _format_rule_title(rule: dict[str, Any], *, command: str, line: str, match: str) -> str:
    template = str(rule.get("title") or "{line}" if line else rule.get("title") or "{match}")
    return template.format(command=command, line=line, match=match)


def _command_title_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"strip_prefixes": (), "rules": (), "max_length": 140}
    return {
        "strip_prefixes": _string_tuple(value.get("strip_prefixes")),
        "rules": tuple(rule for rule in value.get("rules") or () if isinstance(rule, dict)),
        "max_length": value.get("max_length") or 140,
    }


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _truncate_span_title(title: str, *, max_length: int) -> str:
    return title if len(title) <= max_length else f"{title[: max_length - 1].rstrip()}..."


def _compact_text(value: Any) -> str:
    return str(value or "").strip()


def _span_section(name: str, attrs: dict[str, Any]) -> str:
    configured_section = str(attrs.get("span.section") or attrs.get("execution.section") or "").strip()
    if configured_section:
        return configured_section
    if name == "trajectory" and attrs.get("event.type") == "trajectory.complete":
        return "rollout"
    if name in {"trajectory.tool", "llm.request"} or attrs.get("trajectory_id") is not None:
        return "rollout"
    if name.startswith("sandbox."):
        return "sandbox"
    return "eval"


def _configured_exporters(cfg: dict[str, Any], signal: str) -> tuple[str, ...] | None:
    for key in (f"{signal}_exporters", f"{signal}_exporter", "exporters", "exporter"):
        if key in cfg and cfg[key] is not None:
            return _exporter_names(cfg[key])
    return None


def _exporter_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_names = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_names = value
    else:
        raw_names = [value]
    names = tuple(_normalize_exporter_name(name) for name in raw_names if str(name).strip())
    return tuple(name for name in names if name != "none")


def _normalize_exporter_name(value: Any) -> str:
    name = str(value).strip().lower().replace("-", "_")
    return {
        "otlp": "otlp_http",
        "otlp_proto_http": "otlp_http",
        "http": "otlp_http",
        "json": "otlp_json_file",
        "file": "otlp_json_file",
        "local": "otlp_json_file",
        "stdout": "console",
    }.get(name, name)


def _local_trace_export_enabled(
    *,
    output_dir: Path | None,
    export_traces: bool | None,
    trace_exporters: tuple[str, ...] | None,
) -> bool:
    if export_traces is not None:
        return bool(export_traces)
    if trace_exporters is not None:
        return "otlp_json_file" in trace_exporters
    return output_dir is not None


def _live_trace_exporters(cfg: dict[str, Any], configured: tuple[str, ...] | None) -> tuple[str, ...]:
    if configured is not None:
        return tuple(name for name in configured if name != "otlp_json_file")
    if _as_bool(cfg.get("enabled")) and _otel_trace_endpoint(cfg):
        return ("otlp_http",)
    return ()


def _live_metric_exporters(cfg: dict[str, Any], configured: tuple[str, ...] | None) -> tuple[str, ...]:
    if configured is not None:
        return configured
    if _as_bool(cfg.get("enabled")) and _otel_metric_endpoint(cfg):
        return ("otlp_http",)
    return ()


def _build_trace_exporter(name: str, cfg: dict[str, Any]) -> SpanExporter | None:
    if name == "otlp_http":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            return None
        return OTLPSpanExporter(
            endpoint=_otel_trace_endpoint(cfg),
            headers=_otel_headers(cfg, signal="traces"),
            timeout=_otel_timeout(cfg, signal="traces"),
        )
    if name == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()
    raise ValueError(f"Unsupported sandbox trace exporter: {name!r}")


def _build_metric_exporter(name: str, cfg: dict[str, Any]) -> Any:
    if name == "otlp_http":
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        except ImportError:
            return None
        return OTLPMetricExporter(
            endpoint=_otel_metric_endpoint(cfg),
            headers=_otel_headers(cfg, signal="metrics"),
            timeout=_otel_timeout(cfg, signal="metrics"),
        )
    if name == "console":
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

        return ConsoleMetricExporter()
    raise ValueError(f"Unsupported sandbox metric exporter: {name!r}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


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


def _otel_headers(cfg: dict[str, Any], *, signal: str) -> dict[str, str] | None:
    headers = cfg.get(f"{signal}_headers") or cfg.get("headers")
    if not headers:
        return None
    if isinstance(headers, dict):
        return {str(key): str(value) for key, value in headers.items()}
    parsed: dict[str, str] = {}
    for item in str(headers).split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed or None


def _otel_timeout(cfg: dict[str, Any], *, signal: str) -> float | None:
    timeout = cfg.get(f"{signal}_timeout_s") or cfg.get("timeout_s")
    if timeout is None:
        return None
    return float(timeout)


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
    traces_endpoint = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_TRACES_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    )
    metrics_endpoint = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_METRICS_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"
    )
    endpoint = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    traces_exporter = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_TRACES_EXPORTER") or os.environ.get(
        "OTEL_TRACES_EXPORTER"
    )
    metrics_exporter = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_METRICS_EXPORTER") or os.environ.get(
        "OTEL_METRICS_EXPORTER"
    )
    headers = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_HEADERS") or os.environ.get(
        "OTEL_EXPORTER_OTLP_HEADERS"
    )
    traces_headers = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_TRACES_HEADERS") or os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS"
    )
    metrics_headers = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_METRICS_HEADERS") or os.environ.get(
        "OTEL_EXPORTER_OTLP_METRICS_HEADERS"
    )
    timeout_s = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_TIMEOUT_S") or os.environ.get(
        "OTEL_EXPORTER_OTLP_TIMEOUT"
    )
    traces_timeout_s = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_TRACES_TIMEOUT_S") or os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT"
    )
    metrics_timeout_s = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_METRICS_TIMEOUT_S") or os.environ.get(
        "OTEL_EXPORTER_OTLP_METRICS_TIMEOUT"
    )
    command_titles = _json_object_from_env("NEMO_GYM_SANDBOX_OBSERVABILITY_COMMAND_TITLES")
    return {
        "enabled": bool(endpoint or traces_endpoint or metrics_endpoint or traces_exporter or metrics_exporter),
        "service_name": os.environ.get(
            "NEMO_GYM_SANDBOX_OBSERVABILITY_OTEL_SERVICE_NAME",
            os.environ.get("OTEL_SERVICE_NAME", ""),
        ),
        "endpoint": endpoint,
        "traces_endpoint": traces_endpoint,
        "metrics_endpoint": metrics_endpoint,
        "traces_exporter": traces_exporter,
        "metrics_exporter": metrics_exporter,
        "headers": headers,
        "traces_headers": traces_headers,
        "metrics_headers": metrics_headers,
        "timeout_s": timeout_s,
        "traces_timeout_s": traces_timeout_s,
        "metrics_timeout_s": metrics_timeout_s,
        "command_titles": command_titles,
    }


def _json_object_from_env(name: str) -> dict[str, Any] | None:
    value = os.environ.get(name)
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return parsed


def build_recorder_from_config(
    config: dict[str, Any] | None,
    *,
    run_id: str | None = None,
) -> SandboxRecorder | None:
    """Build a recorder from a sandbox observability config."""
    if not isinstance(config, dict) or not config.get("enabled", False):
        return None
    output_dir = config.get("output_dir")
    return SandboxRecorder(
        output_dir=Path(output_dir) if output_dir else None,
        otel=dict(config.get("otel") or {}),
        run_id=run_id,
        run_span_name=config.get("run_span_name") or config.get("job_name"),
        export_traces=config.get("export_traces"),
    )


def build_recorder_from_env() -> SandboxRecorder | None:
    """Build a recorder from eval-job environment variables."""
    output_dir = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_DIR")
    otel = _otel_config_from_env()
    export_traces_env = os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_EXPORT_TRACES")
    export_traces = _as_bool(export_traces_env) if export_traces_env is not None else None
    if not output_dir and not otel.get("enabled"):
        return None
    return SandboxRecorder(
        output_dir=Path(output_dir) if output_dir else None,
        otel=otel,
        run_id=os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_RUN_ID"),
        run_span_name=(
            os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_RUN_SPAN_NAME")
            or os.environ.get("NEMO_GYM_SANDBOX_OBSERVABILITY_JOB_NAME")
            or os.environ.get("JOB_NAME")
            or os.environ.get("KUBE_JOB_NAME")
        ),
        export_traces=export_traces,
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
