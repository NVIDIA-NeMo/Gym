# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-run build: ``rollouts.jsonl`` -> ``metrics/rollouts_timeline.jsonl`` (spans per rollout, tool text tokenised)."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import orjson

from ng_latency.paths import metrics_dir
from ng_latency.project import identity, project_rollout
from ng_latency.spans import TIMELINE_SCHEMA_VERSION
from ng_latency.tokenize import TokenCounter, count_tokens


HEADER_KIND = "header"
CLOCK_NOTES = (
    "Spans carry clock.domain. node:<host> spans share one machine clock per replica (collector, agent server, "
    "model server). sandbox:<id> spans use the sandbox machine's clock. clock_offsets[domain] carries the sandbox "
    "probe's offset_s with its uncertainty_s when the agent measured one, and offset_plus_min_latency_s, the "
    "minimum of (server start - client send) over joined calls, which is the offset plus the smallest network "
    "latency and is reported, never applied. end_kind=derived ends are start + monotonic duration."
)

# Kept under its old name: the analysis modules import it from here.
_identity = identity


def _tool_output_texts(record: dict[str, Any], spans: list[dict[str, Any]]) -> dict[str, str]:
    """span_id -> output text from the harness output: OpenCode tool_call by call_id; Terminus has one observation
    per command batch, so the text belongs to the tool_batch span (counting it per command would multiply it)."""
    response = record.get("response")
    output_items = (response.get("output") or []) if isinstance(response, dict) else []
    by_call_id = {
        item.get("call_id"): str(item.get("output") or "")
        for item in output_items
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    }
    texts: dict[str, str] = {}
    for span in spans:
        if span["name"] == "tool_call":
            call_id = (span.get("ids") or {}).get("tool_call_id")
            if call_id in by_call_id:
                texts[span["span_id"]] = by_call_id[call_id]
        elif span["name"] == "tool_batch":
            index = (span.get("attrs") or {}).get("observation_trajectory_index")
            if isinstance(index, int) and 0 <= index < len(output_items):
                item = output_items[index]
                content = item.get("content") if isinstance(item, dict) else None
                if isinstance(content, list):
                    content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
                if isinstance(content, str):
                    texts[span["span_id"]] = content
    return texts


def _tool_arg_texts(record: dict[str, Any], spans: list[dict[str, Any]]) -> dict[str, str]:
    """span_id -> the tool call's arguments as text: OpenCode from the recorded function_call items, Terminus from
    the command text itself."""
    response = record.get("response")
    output_items = (response.get("output") or []) if isinstance(response, dict) else []
    arguments = {
        item.get("call_id"): (
            item.get("arguments")
            if isinstance(item.get("arguments"), str)
            else json.dumps(item.get("arguments") or {})
        )
        for item in output_items
        if isinstance(item, dict) and item.get("type") == "function_call"
    }
    texts: dict[str, str] = {}
    for span in spans:
        if span["name"] != "tool_call":
            continue
        call_id = (span.get("ids") or {}).get("tool_call_id")
        if call_id in arguments:
            texts[span["span_id"]] = arguments[call_id]
        elif (span.get("attrs") or {}).get("tool") == "terminal":
            texts[span["span_id"]] = str((span.get("attrs") or {}).get("operation") or "")
    return texts


def _producing_server(spans: list[dict[str, Any]], tool: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The server span of the model call that requested this tool: the step it hangs under (OpenCode) or the
    latest call of its session that ended before the tool was requested (Terminus)."""
    by_id = {s["span_id"]: s for s in spans}
    parent = by_id.get(tool.get("parent_span_id") or "")
    while parent is not None:
        server_id = (parent.get("attrs") or {}).get("server_span_id")
        if server_id in by_id:
            return by_id[server_id]
        if parent["name"] in ("model_step", "model_call"):
            break
        parent = by_id.get(parent.get("parent_span_id") or "")
    session = (tool.get("ids") or {}).get("invocation_id")
    requested = (tool.get("attrs") or {}).get("requested_wall_s") or tool["start_wall_s"]
    calls = [
        s
        for s in spans
        if s["name"] == "model_call"
        and (s.get("ids") or {}).get("invocation_id") == session
        and s.get("end_wall_s") is not None
        and s["end_wall_s"] <= float(requested) + 1e-6
        and (s.get("attrs") or {}).get("server_span_id") in by_id
    ]
    if not calls:
        return None
    return by_id[max(calls, key=lambda s: s["end_wall_s"])["attrs"]["server_span_id"]]


def _tokenise_tool_text(record: dict[str, Any], spans: list[dict[str, Any]], token_counter: TokenCounter) -> None:
    """Token counts of tool arguments and tool outputs, from the engine's own tokenizer, after the run."""
    requests: list[tuple[dict[str, Any], str, str]] = []  # (span, key, text)
    arg_texts = _tool_arg_texts(record, spans)
    out_texts = _tool_output_texts(record, spans)
    for span in spans:
        if span["span_id"] in arg_texts:
            requests.append((span, "tool_args", arg_texts[span["span_id"]]))
        if span["span_id"] in out_texts:
            requests.append((span, "tool_output", out_texts[span["span_id"]]))
    if not requests:
        return
    counts = token_counter([text for _, _, text in requests])
    for (span, key, _text), count in zip(requests, counts):
        if count is None:
            continue
        span.setdefault("attrs", {}).setdefault("tokens", {})[key] = count
        span["attrs"]["tokens_source"] = "tokenize_standalone"
        if key == "tool_args":
            server = _producing_server(spans, span)
            if server is not None:
                tokens = server.setdefault("attrs", {}).setdefault("tokens", {})
                tokens["tool_args"] = (tokens.get("tool_args") or 0) + count


def build_rollout_timeline(
    record: dict[str, Any], *, token_counter: Optional[TokenCounter]
) -> Optional[dict[str, Any]]:
    timeline = project_rollout(record)
    if timeline is None:
        return None
    if token_counter is not None:
        _tokenise_tool_text(record, timeline["spans"], token_counter)
    return timeline


def failures_path(run_dir: Path, rollouts_name: str) -> Path:
    """Gym's failure sidecar for this rollouts file: ``<stem>_failures.jsonl`` beside it."""
    rollouts = Path(run_dir) / rollouts_name
    return rollouts.with_name(f"{rollouts.stem}_failures.jsonl")


def count_failures(run_dir: Path, rollouts_name: str) -> dict[str, Any]:
    path = failures_path(run_dir, rollouts_name)
    by_class: Counter = Counter()
    rows = 0
    if path.exists():
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = orjson.loads(line)
                except orjson.JSONDecodeError:
                    by_class["unreadable"] += 1
                    continue
                by_class[str(row.get("_ng_failure_class") or "unknown")] += 1
    return {"path": str(path), "present": path.exists(), "rows": rows, "by_class": dict(sorted(by_class.items()))}


def build_run_timeline(
    run_dir: Path, *, tokenize_url: Optional[str], rollouts_name: str = "rollouts.jsonl"
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    token_counter: Optional[TokenCounter] = (lambda texts: count_tokens(tokenize_url, texts)) if tokenize_url else None
    counts: Counter = Counter()
    kinds: Counter = Counter()
    flags: Counter = Counter()
    gap_histogram: Counter = Counter()
    out_dir = metrics_dir(run_dir)
    out_path = out_dir / "rollouts_timeline.jsonl"
    header = {
        "kind": HEADER_KIND,
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "generated_wall_s": time.time(),
        "clock_notes": CLOCK_NOTES,
        "sources": {"rollouts": str(run_dir / rollouts_name), "tokenize_url": tokenize_url},
    }
    with (run_dir / rollouts_name).open("rb") as source, out_path.open("wb") as sink:
        sink.write(orjson.dumps(header) + b"\n")
        for line in source:
            if not line.strip():
                continue
            record = orjson.loads(line)
            counts["rollouts"] += 1
            built = build_rollout_timeline(record, token_counter=token_counter)
            if built is None:
                gap_histogram["timeline_unavailable"] += 1
                continue
            counts["rollouts_with_timeline"] += 1
            kinds[built["agent_kind"]] += 1
            for flag in built["flags"]:
                flags[flag] += 1
            for span in built["spans"]:
                if span["name"] == "model_call":
                    counts["model_calls"] += 1
                    counts["model_calls_with_server"] += "server_span_id" in (span.get("attrs") or {})
            for gap in built["gaps"]:
                gap_histogram[str(gap.get("code"))] += 1
            sink.write(orjson.dumps(built) + b"\n")
    summary = {
        **{k: counts[k] for k in ("rollouts", "rollouts_with_timeline", "model_calls", "model_calls_with_server")},
        "server_coverage": (counts["model_calls_with_server"] / counts["model_calls"])
        if counts["model_calls"]
        else None,
        "agent_kinds": dict(kinds),
        "flags": dict(sorted(flags.items())),
        "gap_histogram": dict(gap_histogram),
        "collector_failures": count_failures(run_dir, rollouts_name),
        "output": str(out_path),
    }
    (out_dir / "timeline_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
