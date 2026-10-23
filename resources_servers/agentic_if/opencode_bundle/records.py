# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the plugins' JSONL receipts back into the formats the evidence builder consumes.

``model-calls.jsonl`` (written by ``if-boundary.mjs``) holds one JSON object per line:

- ``{"kind": "loaded", "plugin": "if-boundary", "boundary_version", "config_sha256", "plugin_sha256", "pid"}``: the
  load receipt, or ``{"kind": "load_error", "plugin": "if-boundary", "message"}`` when the plugin refused to start;
- ``{"kind": "configured", "provider", "base_url", "denied", "instructions", "system_sha256"}`` from the ``config``
  hook, or ``{"kind": "config_error", "message"}``;
- per observed model call (actor or compaction), in call order:
  ``{"kind": "request", "call", "observation": {source, session_id, assistant_message_id, request, execution_tools}}``
  before the request is sent, then ``{"kind": "response", "call", "response"[, "execution_tools"]}`` and/or
  ``{"kind": "transport_error", "call", "transport_error": ...}`` when the model server reported an error, which
  reached OpenCode unchanged:

  - ``{"phase": "http_rejected", "status"}``: an HTTP error reply, as the 00be1884 gateway recorded it;
  - ``{"phase": "stream_error", "status", "message", "chunks"}``: an error event inside a 200 stream (an
    ``event: error`` field, or data whose top level is an error object) before its terminal event. ``status`` is the
    event's ``status``/``code`` when it names one, else null; ``chunks`` counts the chat-completion chunks OpenCode
    had already received in that stream. Such a call has no ``response``;
- ``{"kind": "issue", "call", "error", "message"}`` for every failure the 00be1884 gateway appended to its ``issues``.

``boundary_records()`` rebuilds ``ModelGateway.observations`` (request time order, the reply merged in) and
``ModelGateway.issues`` (exception type names) from those lines.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue


def parse_jsonl(data: bytes, *, name: str) -> tuple[list[dict[str, JsonValue]], list[str]]:
    """Every JSON object line of ``data``; blank lines are skipped, anything else is a named gap."""
    rows, gaps = [], []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [], [f"{name}_unreadable:not_utf8"]
    # Split on "\n" only: str.splitlines() also breaks at U+2028/U+2029, which JSON strings may hold unescaped.
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            gaps.append(f"{name}_unreadable:line_{number}")
            continue
        if not isinstance(row, dict):
            gaps.append(f"{name}_unreadable:line_{number}")
            continue
        rows.append(row)
    return rows, gaps


@dataclass
class BoundaryRecords:
    """The boundary's records of one sandbox, reassembled."""

    observations: list[dict[str, JsonValue]] = field(default_factory=list)
    """``ModelGateway.observations``: one dict per observed model call, in request order."""
    issues: list[str] = field(default_factory=list)
    """``ModelGateway.issues``: the exception type name of every failed call or stream."""
    loads: list[dict[str, JsonValue]] = field(default_factory=list)
    """The ``loaded`` and ``load_error`` receipts (one per plugin instance)."""
    configurations: list[dict[str, JsonValue]] = field(default_factory=list)
    """The ``configured`` and ``config_error`` receipts (one per plugin instance)."""
    gaps: list[str] = field(default_factory=list)
    """Lines that do not fit the format."""


def boundary_records(rows: Sequence[dict[str, JsonValue]]) -> BoundaryRecords:
    """Reassemble the observations and issues from ``model-calls.jsonl`` rows (see the module docstring)."""
    result = BoundaryRecords()
    calls: dict[str, dict[str, JsonValue]] = {}
    for row in rows:
        kind = row.get("kind")
        if kind in {"loaded", "load_error"}:
            result.loads.append(row)
        elif kind in {"configured", "config_error"}:
            result.configurations.append(row)
        elif kind == "issue":
            error = row.get("error")
            result.issues.append(error if isinstance(error, str) and error else "UnknownError")
        elif kind == "request":
            identity, observation = row.get("call"), row.get("observation")
            if not isinstance(identity, str) or identity in calls or not isinstance(observation, dict):
                result.gaps.append("model_call_records_invalid:request")
                continue
            calls[identity] = dict(observation)
            result.observations.append(calls[identity])
        elif kind in {"response", "transport_error"}:
            observation = calls.get(row.get("call"))
            if observation is None or kind not in row:
                result.gaps.append(f"model_call_records_invalid:{kind}")
                continue
            observation[kind] = row[kind]
            if kind == "response" and "execution_tools" in row:
                observation["execution_tools"] = row["execution_tools"]
        else:
            result.gaps.append("model_call_records_invalid:kind")
    return result
