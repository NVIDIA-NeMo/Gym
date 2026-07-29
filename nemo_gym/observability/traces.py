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

"""OTLP JSON and Chrome trace export for NeMo Gym observability."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from nemo_gym.observability.events import stable_hash
from nemo_gym.observability.summary import load_jsonl


_SCOPE_NAME = "nemo_gym.observability"
_SCOPE_VERSION = "1"
_SEMANTIC_LANES = {
    "foreground": (1, "Foreground tool / agent"),
    "llm": (2, "LLM inference"),
    "background": (3, "Background tool process"),
}
_SEMANTIC_TOOL_NAMES = {"trajectory.tool", "agent.tool"}
_TRAJECTORY_TERMINAL_NAMES = {"trajectory.complete", "trajectory.masked"}


def export_trace_artifacts(
    output_dir: Path,
    *,
    service_name: str = "nemo-gym-eval",
    run_id: str | None = None,
) -> dict[str, str]:
    """Export OTLP JSON and Chrome trace files under ``output_dir/traces/``.

    The OTLP JSON file is the canonical interchange artifact for sending to
    Honeycomb, Grafana Tempo, Jaeger, or any OTLP-compatible backend.
    The Chrome trace file can be opened directly in Perfetto or chrome://tracing.
    """
    events = load_jsonl(output_dir / "events.jsonl")
    if not events:
        return {}

    spans, instant_events = _build_spans(events, run_id=run_id)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    otlp_path = traces_dir / "otel_traces.json"
    otlp_payload = _otlp_payload(
        spans,
        service_name=service_name,
        run_id=run_id,
    )
    otlp_path.write_text(
        json.dumps(otlp_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    chrome_path = traces_dir / "chrome_trace.json"
    chrome_payload = _chrome_trace_payload(spans, instant_events)
    chrome_path.write_text(
        json.dumps(chrome_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "otlp_json": str(otlp_path),
        "chrome_trace": str(chrome_path),
    }


def _build_spans(
    events: list[dict[str, Any]],
    *,
    run_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    starts: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    starts_by_span_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    spans = []
    instant_events = []

    for index, event in enumerate(sorted(events, key=lambda row: float(row.get("timestamp_unix_s") or 0.0))):
        event_type = str(event.get("event_type") or "event")
        name = str(event.get("name") or "unknown")
        attrs = dict(event.get("attributes") or {})
        timestamp = _event_timestamp(event)
        if timestamp is None:
            continue

        if event_type == "span_start":
            starts[_span_key(name, attrs)].append(event)
            span_id = event.get("span_id")
            if isinstance(span_id, str):
                starts_by_span_id[span_id].append(event)
            continue

        if event_type == "span_end":
            start_event = _pop_matching_start(starts, starts_by_span_id, name, attrs, event)
            start_attrs = dict(start_event.get("attributes") or {}) if start_event else {}
            merged_attrs = {**start_attrs, **attrs}
            trace_id, span_id, parent_span_id = _event_trace_ids(event, start_event)
            start_timestamp = _event_timestamp(start_event) if start_event else None
            duration_s = merged_attrs.get("duration_s")
            if start_timestamp is None and isinstance(duration_s, (int, float)):
                start_timestamp = timestamp - float(duration_s)
            if start_timestamp is None:
                start_timestamp = timestamp
            end_timestamp = max(timestamp, start_timestamp)
            if not isinstance(duration_s, (int, float)):
                merged_attrs["duration_s"] = max(end_timestamp - start_timestamp, 0.0)
            spans.append(
                _span_record(
                    name=name,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    attrs=merged_attrs,
                    event_type=event_type,
                    index=index,
                    run_id=run_id,
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                )
            )
            continue

        instant = _span_record(
            name=name,
            start_timestamp=timestamp,
            end_timestamp=timestamp,
            attrs=attrs,
            event_type=event_type,
            index=index,
            run_id=run_id,
            trace_id=_string_value(event.get("trace_id")),
            span_id=_string_value(event.get("span_id")),
            parent_span_id=_string_value(event.get("parent_span_id")),
        )
        instant_events.append(instant)

    trace_spans = _add_trajectory_root_spans(
        spans + instant_events,
        events,
        run_id=run_id,
    )
    return trace_spans, instant_events


def _span_record(
    *,
    name: str,
    start_timestamp: float,
    end_timestamp: float,
    attrs: dict[str, Any],
    event_type: str,
    index: int,
    run_id: str | None,
    trace_id: str | None,
    span_id: str | None,
    parent_span_id: str | None,
) -> dict[str, Any]:
    trace_group = _trace_group(attrs, run_id)
    identity = f"{trace_group}:{name}:{start_timestamp:.9f}:{end_timestamp:.9f}:{index}"
    span_attrs = _standardized_attributes(name, attrs, run_id=run_id)
    record = {
        "trace_id": trace_id or stable_hash(trace_group, length=32),
        "span_id": span_id or stable_hash(identity, length=16),
        "name": name,
        "kind": _span_kind(name),
        "event_type": event_type,
        "start_time_unix_nano": _to_unix_nano(start_timestamp),
        "end_time_unix_nano": _to_unix_nano(end_timestamp),
        "start_time_unix_s": start_timestamp,
        "end_time_unix_s": end_timestamp,
        "attributes": {
            "event.type": event_type,
            **span_attrs,
        },
    }
    if parent_span_id:
        record["parent_span_id"] = parent_span_id
    return record


def _add_trajectory_root_spans(
    spans: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    run_id: str | None,
) -> list[dict[str, Any]]:
    spans_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    terminal_events: dict[str, dict[str, Any]] = {}
    for span in spans:
        trajectory_id = _trajectory_id(span["attributes"])
        if trajectory_id:
            spans_by_trajectory[trajectory_id].append(span)

    for event in events:
        name = str(event.get("name") or "")
        if name not in _TRAJECTORY_TERMINAL_NAMES:
            continue
        attrs = dict(event.get("attributes") or {})
        trajectory_id = _trajectory_id(attrs)
        if trajectory_id:
            terminal_events[trajectory_id] = event

    root_spans = []
    for trajectory_id, child_spans in sorted(spans_by_trajectory.items()):
        trace_id = stable_hash(f"{run_id or 'run'}:{trajectory_id}", length=32)
        root_span_id = stable_hash(f"{trace_id}:trajectory.root", length=16)
        child_span_ids = {span["span_id"] for span in child_spans}
        for span in child_spans:
            span["trace_id"] = trace_id
            parent_span_id = span.get("parent_span_id")
            if not parent_span_id or parent_span_id not in child_span_ids:
                span["parent_span_id"] = root_span_id

        start_time = min(float(span["start_time_unix_s"]) for span in child_spans)
        end_time = max(float(span["end_time_unix_s"]) for span in child_spans)
        terminal_event = terminal_events.get(trajectory_id)
        terminal_attrs = dict(terminal_event.get("attributes") or {}) if terminal_event else {}
        terminal_time = _event_timestamp(terminal_event)
        if terminal_time is not None:
            end_time = max(end_time, terminal_time)
            duration_s = terminal_attrs.get("duration_s")
            if isinstance(duration_s, (int, float)) and duration_s >= 0:
                start_time = min(start_time, terminal_time - float(duration_s))

        root_attrs = _trajectory_root_attributes(
            trajectory_id,
            terminal_attrs,
            run_id=run_id,
        )
        root_spans.append(
            {
                "trace_id": trace_id,
                "span_id": root_span_id,
                "name": "trajectory",
                "kind": "SPAN_KIND_INTERNAL",
                "event_type": "synthetic_root",
                "start_time_unix_nano": _to_unix_nano(start_time),
                "end_time_unix_nano": _to_unix_nano(max(end_time, start_time)),
                "start_time_unix_s": start_time,
                "end_time_unix_s": max(end_time, start_time),
                "attributes": root_attrs,
            }
        )

    return root_spans + spans


def _trajectory_root_attributes(
    trajectory_id: str,
    terminal_attrs: dict[str, Any],
    *,
    run_id: str | None,
) -> dict[str, Any]:
    attrs = {
        "event.type": "synthetic_root",
        "phase": "trajectory",
        "status": str(terminal_attrs.get("status") or "ok"),
        "trajectory_id": trajectory_id,
        "nemo_gym.trace.role": "trajectory_root",
        "nemo_gym.trajectory_id": trajectory_id,
        "nemo_gym.trajectory_id_hash": stable_hash(trajectory_id, length=16),
        "wandb.display_name": f"trajectory {trajectory_id}",
        "wandb.is_turn": True,
        "wandb.thread_id": _wandb_thread_id(trajectory_id, run_id),
        "weave.span.kind": "agent",
    }
    if run_id:
        attrs["nemo_gym.run_id"] = run_id
        attrs["wandb.wb_run_id"] = run_id
    for key in (
        "reward",
        "stop_reason",
        "duration_s",
        "loss_multiplier",
        "attempt_idx",
        "harness",
        "dataset_alias",
    ):
        if terminal_attrs.get(key) is not None:
            attrs[key] = terminal_attrs[key]
    attrs["input.value"] = _trajectory_input_value(trajectory_id, run_id)
    attrs["output.value"] = _trajectory_output_value(attrs)
    return attrs


def _trajectory_id(attrs: dict[str, Any]) -> str | None:
    value = attrs.get("trajectory_id") or attrs.get("trial_name")
    return str(value) if value else None


def _wandb_thread_id(trajectory_id: str, run_id: str | None) -> str:
    return f"{run_id or 'run'}:{trajectory_id}"


def _trajectory_input_value(trajectory_id: str, run_id: str | None) -> str:
    if run_id:
        return f"trajectory_id={trajectory_id}; run_id={run_id}"
    return f"trajectory_id={trajectory_id}"


def _trajectory_output_value(attrs: dict[str, Any]) -> str:
    parts = []
    for key in ("reward", "stop_reason", "duration_s", "loss_multiplier"):
        if key in attrs:
            parts.append(f"{key}={attrs[key]}")
    return "; ".join(parts) if parts else "trajectory recorded"


def _span_key(name: str, attrs: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        name,
        str(attrs.get("trajectory_id") or attrs.get("trial_name") or ""),
        str(attrs.get("sandbox_id") or ""),
        str(attrs.get("phase") or ""),
    )


def _pop_matching_start(
    starts: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    starts_by_span_id: dict[str, list[dict[str, Any]]],
    name: str,
    attrs: dict[str, Any],
    end_event: dict[str, Any],
) -> dict[str, Any] | None:
    span_id = end_event.get("span_id")
    if isinstance(span_id, str) and starts_by_span_id.get(span_id):
        start_event = starts_by_span_id[span_id].pop()
        for stack in starts.values():
            if start_event in stack:
                stack.remove(start_event)
                break
        return start_event

    key = _span_key(name, attrs)
    if starts.get(key):
        return starts[key].pop()

    trajectory_id = str(attrs.get("trajectory_id") or attrs.get("trial_name") or "")
    sandbox_id = str(attrs.get("sandbox_id") or "")
    for candidate, stack in starts.items():
        if not stack:
            continue
        candidate_name, candidate_trajectory_id, candidate_sandbox_id, _phase = candidate
        if candidate_name == name and candidate_trajectory_id == trajectory_id and candidate_sandbox_id == sandbox_id:
            return stack.pop()
    return None


def _event_trace_ids(
    event: dict[str, Any],
    start_event: dict[str, Any] | None,
) -> tuple[str | None, str | None, str | None]:
    start_event = start_event or {}
    return (
        _string_value(event.get("trace_id") or start_event.get("trace_id")),
        _string_value(event.get("span_id") or start_event.get("span_id")),
        _string_value(event.get("parent_span_id") or start_event.get("parent_span_id")),
    )


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _trace_group(attrs: dict[str, Any], run_id: str | None) -> str:
    trajectory_id = attrs.get("trajectory_id") or attrs.get("trial_name")
    if trajectory_id:
        return f"{run_id or 'run'}:{trajectory_id}"
    sandbox_id = attrs.get("sandbox_id")
    if sandbox_id:
        return f"{run_id or 'run'}:sandbox:{sandbox_id}"
    return run_id or "nemo-gym-observability-run"


def _standardized_attributes(
    name: str,
    attrs: dict[str, Any],
    *,
    run_id: str | None,
) -> dict[str, Any]:
    standardized = dict(attrs)
    if run_id:
        standardized.setdefault("nemo_gym.run_id", run_id)

    trajectory_id = _trajectory_id(standardized)
    if trajectory_id:
        standardized.setdefault("nemo_gym.trajectory_id", trajectory_id)
        standardized.setdefault(
            "nemo_gym.trajectory_id_hash",
            stable_hash(trajectory_id, length=16),
        )
        standardized.setdefault("wandb.thread_id", _wandb_thread_id(trajectory_id, run_id))
        standardized.setdefault("wandb.is_turn", False)

    sandbox_id = standardized.get("sandbox_id")
    if sandbox_id:
        standardized.setdefault(
            "nemo_gym.sandbox_id_hash",
            stable_hash(str(sandbox_id), length=16),
        )

    _copy_attr(standardized, "phase", "nemo_gym.phase")
    _copy_attr(standardized, "command_class", "nemo_gym.command.class")
    _copy_attr(standardized, "command_hash", "nemo_gym.command.hash")
    _copy_attr(standardized, "tool_name", "nemo_gym.tool.name")
    _copy_attr(standardized, "return_code", "nemo_gym.tool.return_code")
    _copy_attr(standardized, "duration_s", "nemo_gym.duration_s")
    if run_id:
        standardized.setdefault("wandb.wb_run_id", run_id)
    if standardized.get("return_code") not in (None, 0, "0"):
        standardized.setdefault("nemo_gym.tool.status", "nonzero_return")

    if name == "llm.request":
        standardized.setdefault("gen_ai.system", "nemo_gym.policy_proxy")
        standardized.setdefault("weave.span.kind", "llm")
        upstream_api = str(standardized.get("upstream_api") or "chat")
        standardized.setdefault("gen_ai.operation.name", upstream_api)
        model_name = (
            standardized.get("model") or standardized.get("model_name") or standardized.get("upstream_model_name")
        )
        if model_name is not None:
            standardized.setdefault("gen_ai.request.model", str(model_name))
        _copy_attr(standardized, "prompt_tokens", "gen_ai.usage.input_tokens")
        _copy_attr(standardized, "completion_tokens", "gen_ai.usage.output_tokens")
        _copy_attr(standardized, "total_tokens", "gen_ai.usage.total_tokens")
    elif name in _SEMANTIC_TOOL_NAMES or name.startswith("sandbox."):
        standardized.setdefault("weave.span.kind", "tool")
    return standardized


def _copy_attr(attrs: dict[str, Any], source: str, target: str) -> None:
    if source in attrs and attrs[source] is not None:
        attrs.setdefault(target, attrs[source])


def _span_kind(name: str) -> str:
    if name == "llm.request":
        return "SPAN_KIND_CLIENT"
    return "SPAN_KIND_INTERNAL"


def _event_timestamp(event: dict[str, Any] | None) -> float | None:
    if event is None:
        return None
    timestamp = event.get("timestamp_unix_s")
    return float(timestamp) if isinstance(timestamp, (int, float)) else None


def _to_unix_nano(timestamp_unix_s: float) -> str:
    return str(max(0, int(timestamp_unix_s * 1_000_000_000)))


def _otlp_payload(
    spans: list[dict[str, Any]],
    *,
    service_name: str,
    run_id: str | None,
) -> dict[str, Any]:
    resource_attributes = [
        _attribute("service.name", service_name),
        _attribute("telemetry.sdk.language", "python"),
        _attribute("telemetry.sdk.name", "opentelemetry"),
    ]
    if run_id:
        resource_attributes.append(_attribute("nemo_gym.run_id", run_id))

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": resource_attributes,
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": _SCOPE_NAME,
                            "version": _SCOPE_VERSION,
                        },
                        "spans": [_otlp_span(span) for span in spans],
                    }
                ],
            }
        ]
    }


def _otlp_span(span: dict[str, Any]) -> dict[str, Any]:
    attrs = span["attributes"]
    status_code = _otlp_status_code(span["name"], attrs)
    row = {
        "traceId": span["trace_id"],
        "spanId": span["span_id"],
        "name": span["name"],
        "kind": span.get("kind") or "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": span["start_time_unix_nano"],
        "endTimeUnixNano": span["end_time_unix_nano"],
        "attributes": [_attribute(key, value) for key, value in sorted(attrs.items()) if value is not None],
        "status": {"code": status_code},
    }
    if span.get("parent_span_id"):
        row["parentSpanId"] = span["parent_span_id"]
    return row


def _otlp_status_code(name: str, attrs: dict[str, Any]) -> str:
    if name in _SEMANTIC_TOOL_NAMES and attrs.get("return_code") not in (None, 0, "0"):
        return "STATUS_CODE_OK"
    status = str(attrs.get("status") or "").lower()
    if status == "error" or attrs.get("error_type") or attrs.get("exception_type"):
        return "STATUS_CODE_ERROR"
    return "STATUS_CODE_OK"


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


def _chrome_trace_payload(
    spans: list[dict[str, Any]],
    instant_events: list[dict[str, Any]],
) -> dict[str, Any]:
    semantic_spans = _semantic_trace_spans(spans)
    process_ids: dict[str, int] = {}
    thread_names: set[tuple[int, int]] = set()
    trace_events = []

    for span in semantic_spans:
        attrs = span["attributes"]
        scope = _chrome_scope(attrs)
        if scope not in process_ids:
            process_id = len(process_ids) + 1
            process_ids[scope] = process_id
            trace_events.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": process_id,
                    "tid": 0,
                    "args": {"name": scope},
                }
            )
        else:
            process_id = process_ids[scope]

        lane_id, lane_name = _semantic_lane(span)
        thread_key = (process_id, lane_id)
        if thread_key not in thread_names:
            thread_names.add(thread_key)
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": process_id,
                    "tid": lane_id,
                    "args": {"name": lane_name},
                }
            )

        duration_us = max(
            int((span["end_time_unix_s"] - span["start_time_unix_s"]) * 1_000_000),
            1,
        )
        trace_events.append(
            {
                "name": _semantic_display_name(span),
                "cat": _semantic_category(span),
                "ph": "X",
                "ts": int(span["start_time_unix_s"] * 1_000_000),
                "dur": duration_us,
                "pid": process_id,
                "tid": lane_id,
                "args": attrs,
            }
        )

    instant_ids = {event["span_id"] for event in instant_events}
    for event in trace_events:
        if event.get("ph") != "X":
            continue
        if event["args"].get("event.type") != "span_end" and event["dur"] == 1:
            event["ph"] = "i"
            event["s"] = "t"
            event.pop("dur", None)

    return {
        "displayTimeUnit": "ms",
        "traceEvents": trace_events,
        "metadata": {
            "format": "chrome_trace_event",
            "instant_event_count": len(instant_ids),
            "span_count": len(semantic_spans),
            "trace_view": "trajectory_semantic",
        },
    }


def _semantic_trace_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes_with_tools = {_chrome_scope(span["attributes"]) for span in spans if span["name"] in _SEMANTIC_TOOL_NAMES}
    has_trajectory_tools = bool(scopes_with_tools)
    semantic_spans = []
    for span in spans:
        attrs = span["attributes"]
        if attrs.get("event.type") == "process":
            continue
        if attrs.get("nemo_gym.trace.role") == "trajectory_root":
            continue
        name = span["name"]
        scope = _chrome_scope(attrs)
        if has_trajectory_tools and scope == "run":
            continue
        if name in _SEMANTIC_TOOL_NAMES or name == "llm.request":
            semantic_spans.append(span)
            continue
        if scope in scopes_with_tools and not (
            name.startswith("trajectory.") or name == "trajectory.background_process"
        ):
            continue
        semantic_spans.append(span)
    return semantic_spans


def _semantic_lane(span: dict[str, Any]) -> tuple[int, str]:
    attrs = span["attributes"]
    name = span["name"]
    if name == "llm.request" or attrs.get("phase") == "llm":
        return _SEMANTIC_LANES["llm"]
    if name == "trajectory.background_process" or attrs.get("phase") == "background":
        return _SEMANTIC_LANES["background"]
    return _SEMANTIC_LANES["foreground"]


def _semantic_category(span: dict[str, Any]) -> str:
    attrs = span["attributes"]
    if span["name"] == "llm.request" or attrs.get("phase") == "llm":
        return "llm"
    if attrs.get("phase") == "background":
        return "background"
    return str(attrs.get("command_class") or attrs.get("phase") or "event")


def _semantic_display_name(span: dict[str, Any]) -> str:
    if span["name"] not in _SEMANTIC_TOOL_NAMES:
        return span["name"]
    attrs = span["attributes"]
    command_class = str(attrs.get("command_class") or "tool")
    tool_name = str(attrs.get("tool_name") or "tool")
    return f"{tool_name}:{command_class}"


def _chrome_scope(attrs: dict[str, Any]) -> str:
    return str(attrs.get("trajectory_id") or attrs.get("trial_name") or attrs.get("sandbox_id") or "run")
