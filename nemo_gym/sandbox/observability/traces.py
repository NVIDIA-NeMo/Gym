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

"""OpenTelemetry SDK trace artifact exporters for sandbox observability."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind, StatusCode


SCOPE_NAME = "nemo_gym.sandbox.observability"
SCOPE_VERSION = "1"


class JsonSpanExporter(SpanExporter):
    """OpenTelemetry span exporter that keeps finished spans for local artifacts."""

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            if self._shutdown:
                return SpanExportResult.FAILURE
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True

    def finished_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)


def export_trace_artifacts(
    output_dir: Path,
    *,
    spans: Sequence[ReadableSpan],
    service_name_strategy: str | None = "span_section",
) -> dict[str, str]:
    """Export SDK-finished spans as an OTLP-shaped JSON trace artifact."""
    if not spans:
        return {}

    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    otlp_path = traces_dir / "otel_traces.json"
    otlp_path.write_text(
        json.dumps(_otlp_payload(spans, service_name_strategy=service_name_strategy), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"otlp_json": str(otlp_path)}


def _otlp_payload(spans: Sequence[ReadableSpan], *, service_name_strategy: str | None) -> dict[str, Any]:
    resource_groups: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}

    for span in spans:
        resource_attributes = _resource_attributes(span, service_name_strategy=service_name_strategy)
        resource_key = tuple(sorted((key, str(value)) for key, value in resource_attributes.items()))
        resource_group = resource_groups.setdefault(
            resource_key,
            {
                "resource": {
                    "attributes": [_attribute(key, value) for key, value in sorted(resource_attributes.items())]
                },
                "scopeSpans": {},
            },
        )
        scope = span.instrumentation_scope
        scope_key = (
            getattr(scope, "name", None) or SCOPE_NAME,
            getattr(scope, "version", None) or SCOPE_VERSION,
        )
        scope_group = resource_group["scopeSpans"].setdefault(
            scope_key,
            {
                "scope": {
                    "name": scope_key[0],
                    "version": scope_key[1],
                },
                "spans": [],
            },
        )
        scope_group["spans"].append(_otlp_span(span))

    resource_spans = []
    for group in resource_groups.values():
        scope_spans = list(group["scopeSpans"].values())
        resource_spans.append({"resource": group["resource"], "scopeSpans": scope_spans})
    return {"resourceSpans": resource_spans}


def _resource_attributes(span: ReadableSpan, *, service_name_strategy: str | None) -> dict[str, Any]:
    attrs = dict(span.resource.attributes)
    if _uses_visual_service_lanes(service_name_strategy):
        service_name = _visual_service_name(span)
        if service_name:
            original_service_name = attrs.get("service.name")
            if original_service_name:
                attrs.setdefault("service.original_name", original_service_name)
            attrs["service.name"] = service_name
    return attrs


def _otlp_span(span: ReadableSpan) -> dict[str, Any]:
    row: dict[str, Any] = {
        "traceId": _trace_id(span.context.trace_id),
        "spanId": _span_id(span.context.span_id),
        "name": span.name,
        "kind": _span_kind(span.kind),
        "startTimeUnixNano": str(span.start_time or 0),
        "endTimeUnixNano": str(span.end_time or span.start_time or 0),
        "attributes": [
            _attribute(key, value) for key, value in sorted((span.attributes or {}).items()) if value is not None
        ],
        "events": [_otlp_event(event) for event in span.events],
        "status": _otlp_status(span),
    }
    if span.parent is not None and span.parent.span_id:
        row["parentSpanId"] = _span_id(span.parent.span_id)
    return row


def _otlp_event(event: Any) -> dict[str, Any]:
    return {
        "name": event.name,
        "timeUnixNano": str(event.timestamp),
        "attributes": [
            _attribute(key, value) for key, value in sorted((event.attributes or {}).items()) if value is not None
        ],
    }


def _otlp_status(span: ReadableSpan) -> dict[str, str]:
    status_code = span.status.status_code
    if status_code == StatusCode.ERROR:
        code = "STATUS_CODE_ERROR"
    elif status_code == StatusCode.OK:
        code = "STATUS_CODE_OK"
    else:
        code = "STATUS_CODE_UNSET"
    status = {"code": code}
    if span.status.description:
        status["message"] = span.status.description
    return status


def _span_kind(kind: SpanKind) -> str:
    return f"SPAN_KIND_{kind.name}"


def _trace_id(value: int) -> str:
    return f"{value:032x}"


def _uses_visual_service_lanes(strategy: str | None) -> bool:
    return str(strategy or "").strip().lower() in {"span_section", "visual", "visual_lanes", "service_lanes"}


def _visual_service_name(span: ReadableSpan) -> str | None:
    attrs = dict(span.attributes or {})
    operation_name = str(attrs.get("operation.name") or span.name or "")
    span_role = str(attrs.get("span.role") or "")

    if operation_name == "sandbox.run" or span_role == "eval.run":
        return "nemo-gym.eval"
    if operation_name == "trajectory" or span_role.endswith(".trajectory"):
        section = str(attrs.get("span.section") or "rollout")
        return f"nemo-gym.{section}"
    if operation_name == "llm.request":
        return "llm.request"
    if operation_name in {
        "sandbox.start",
        "sandbox.start_batch",
        "sandbox.create",
        "sandbox.create_api",
        "sandbox.create_probe",
    }:
        return "sandbox.create"
    if attrs.get("span.section") == "verifier" and operation_name in {"trajectory.tool", "sandbox.exec"}:
        return "verifier.exec"
    if operation_name in {"trajectory.tool", "sandbox.exec"}:
        return "sandbox.exec"
    if operation_name.startswith("sandbox.diagnostic."):
        return "sandbox.diagnostic"
    if operation_name in {"sandbox.cleanup", "sandbox.close", "sandbox.delete"}:
        return "sandbox.cleanup"
    if operation_name in {"sandbox.read_file", "sandbox.write_file", "sandbox.upload_file", "sandbox.download_file"}:
        return "sandbox.io"

    section = attrs.get("span.section")
    if section in {"eval", "rollout", "verifier"}:
        return f"nemo-gym.{section}"
    if section == "sandbox":
        return "sandbox"
    return None


def _span_id(value: int) -> str:
    return f"{value:016x}"


def _attribute(key: str, value: Any) -> dict[str, Any]:
    return {
        "key": key,
        "value": _otel_value(value),
    }


def _otel_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_otel_value(item) for item in value]}}
    return {"stringValue": str(value)}
