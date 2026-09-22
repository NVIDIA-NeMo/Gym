# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming native Gym reader with explicit, loss-detecting payload joins."""

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator


PAYLOADS = ("request", "response", "request_raw", "response_raw")


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON number")


def json_rows(path: Path) -> Iterator[tuple[int, dict]]:
    """Read JSONL strictly; malformed/truncated rows are checker errors."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line, parse_constant=_reject_constant)
                except (ValueError, RecursionError) as exc:
                    raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name}:{number}: expected an object")
                yield number, value


def digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def hydrate_record(record: dict, *, capture_dir: Path | None = None) -> dict:
    """Restore payloads by exact model_call_id, never order or content hash.

    The native collector moves bodies from capture.calls into ng_trajectory.
    An optional capture directory provides the retained transport exchanges.
    Conflicts are preserved as reader failures instead of being overwritten.
    """
    result = copy.deepcopy(record)
    capture = result.get("ng_model_call_capture") or {}
    calls = capture.get("calls") or []
    trajectory = result.get("ng_trajectory") or {}
    issues = result.setdefault("_capability_reader_issues", [])
    by_id: dict[str, dict] = {}
    for call in calls:
        if isinstance(call, dict) and isinstance(call.get("model_call_id"), str):
            call_id = call["model_call_id"]
            if call_id in by_id:
                issues.append("duplicate model call identity")
            by_id[call_id] = call

    def merge(items: list[dict], *, sidecar: bool) -> None:
        seen: set[str] = set()
        for item in items:
            call_id = item.get("model_call_id")
            if not isinstance(call_id, str) or call_id not in by_id or call_id in seen:
                issues.append("unmatched or duplicate payload identity")
            else:
                seen.add(call_id)
                call = by_id[call_id]
                for key in PAYLOADS:
                    value = item.get(key)
                    if value is not None:
                        if call.get(key) is not None and call[key] != value:
                            issues.append("conflicting payload representations")
                        else:
                            call[key] = value
                metadata = item if sidecar else item.get("response_metadata", {})
                for key in (
                    "model_ref",
                    "response_id",
                    "status_code",
                    "dialect",
                    "error_category",
                ):
                    if metadata.get(key) is not None and call.get(key) != metadata[key]:
                        issues.append("conflicting model-call metadata")
                if not sidecar:
                    stats = item.get("token_stats") or {}
                    for canonical, captured in (
                        ("prompt_tokens", "tokens_in"),
                        ("completion_tokens", "tokens_out"),
                        ("reasoning_tokens", "tokens_reasoning"),
                        ("total_tokens", "tokens_total"),
                        ("cached_tokens", "cached_tokens"),
                    ):
                        if canonical in stats and stats[canonical] != call.get(captured):
                            issues.append("trajectory and capture token counts disagree")
        if sidecar and seen != set(by_id):
            issues.append("capture sidecar and rollout have different identity sets")

    merge(trajectory.get("model_calls") or [], sidecar=False)
    if capture_dir is not None:
        rollout_id = capture.get("rollout_id")
        if not isinstance(rollout_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", rollout_id):
            issues.append("unsafe or missing capture identity")
        else:
            path = capture_dir / f"{rollout_id}.capture.jsonl"
            if path.is_file():
                merge([row for _, row in json_rows(path)], sidecar=True)
            else:
                issues.append("requested capture sidecar is missing")
            if (capture_dir / f"{rollout_id}.capture.incomplete").exists():
                issues.append("capture is marked incomplete")
    return result
