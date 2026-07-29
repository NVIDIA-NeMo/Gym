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

"""Event schema and command classifiers for NeMo Gym observability."""

from __future__ import annotations

import hashlib
import re
from typing import Any, NotRequired, TypedDict


SCHEMA_VERSION = 1


class SandboxEvent(TypedDict):
    """Versioned event written to ``observability/events.jsonl``."""

    schema_version: int
    timestamp_unix_s: float
    monotonic_s: float | None
    elapsed_time_s: float
    event_type: str
    name: str
    trace_id: NotRequired[str]
    span_id: NotRequired[str]
    parent_span_id: NotRequired[str]
    attributes: NotRequired[dict[str, Any]]


class ResourceSample(TypedDict):
    """Versioned resource sample written to ``resource_samples.jsonl``."""

    schema_version: int
    timestamp_unix_s: float
    monotonic_s: float | None
    elapsed_time_s: float
    sandbox_id: str | None
    cpu_utilization: NotRequired[float | None]
    cpu_usage_seconds_total: NotRequired[float | None]
    memory_usage_bytes: NotRequired[int | None]
    process_count: NotRequired[int | None]
    processes: NotRequired[list[dict[str, Any]]]
    attributes: NotRequired[dict[str, Any]]


_READ_EDIT_RE = re.compile(
    r"\b("
    r"cat|sed|grep|rg|find|ls|head|tail|less|awk|cut|sort|uniq|wc|"
    r"git\s+(diff|status|show|grep|ls-files)|"
    r"apply_patch|python\s+-m\s+pip|touch|mkdir|mv|cp|rm|chmod|chown|tee"
    r")\b"
)
_EDIT_RE = re.compile(r"(>|>>|\bsed\s+-i\b|\bperl\s+-pi\b|\bpatch\b|\bgit\s+apply\b)")
_BUILD_RE = re.compile(
    r"\b("
    r"make|cmake|ninja|gcc|g\+\+|clang|cargo|go\s+build|"
    r"npm\s+(install|ci|run\s+build)|yarn\s+(install|build)|"
    r"pnpm\s+(install|build)|pip\s+install|uv\s+(sync|pip|run)|"
    r"pytest|tox|maturin|setup\.py|build"
    r")\b"
)
_SLEEP_RE = re.compile(r"\b(sleep|watch|tail\s+-f|kubectl\s+wait)\b|\bwhile\b.*\bsleep\b")
_COMPUTE_RE = re.compile(r"\b(python|python3|node|bash|sh|ruby|java|Rscript)\b")


def stable_hash(value: str, *, length: int = 12) -> str:
    """Return a short stable hash for sensitive or high-cardinality values."""
    return hashlib.blake2s(value.encode("utf-8"), digest_size=16).hexdigest()[:length]


def classify_command(command: str | None) -> str:
    """Classify a shell command for timeline visualization.

    Returns one of: sleep_poll, build, read_write_edit, foreground_compute, other_bash.
    """
    if not command:
        return "other_bash"
    normalized = " ".join(command.strip().lower().split())
    if _SLEEP_RE.search(normalized):
        return "sleep_poll"
    if _BUILD_RE.search(normalized):
        return "build"
    if _EDIT_RE.search(normalized) or _READ_EDIT_RE.search(normalized):
        return "read_write_edit"
    if _COMPUTE_RE.search(normalized):
        return "foreground_compute"
    return "other_bash"


def command_attributes(
    command: str | None,
    *,
    include_command_text: bool,
) -> dict[str, Any]:
    """Build low-risk command attributes for event records."""
    attrs: dict[str, Any] = {"command_class": classify_command(command)}
    if command is None:
        return attrs
    attrs["command_hash"] = stable_hash(command)
    if include_command_text:
        attrs["command"] = command
    else:
        attrs["command_redacted"] = True
    return attrs


def safe_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Return JSON-friendly event attributes."""
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
        elif isinstance(value, dict):
            safe[key] = {
                str(child_key): (
                    child_value
                    if isinstance(child_value, (str, int, float, bool)) or child_value is None
                    else str(child_value)
                )
                for child_key, child_value in value.items()
            }
        else:
            safe[key] = str(value)
    return safe
