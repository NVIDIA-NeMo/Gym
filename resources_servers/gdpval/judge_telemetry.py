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
"""Optional structured telemetry for GDPVal judge requests.

The event log intentionally excludes prompts, attachment contents, base64,
credentials, and absolute paths. Each resources-server process writes its own
JSONL file so deployments on shared filesystems do not depend on cross-process
append semantics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1
EVENT_FILE_GLOB = "judge-events.*.jsonl"
_LABEL_RE = re.compile(r"^\n(?P<name>[^\n]+):\n$")
_STRUCTURED_LABEL_RE = re.compile(r"^\n(?P<name>[^\n]+) \(structured spreadsheet cells\):\n")
_STATUS_RE = re.compile(r"(?:error\s+code|status(?:\s+code)?)\D{0,12}(?P<status>[1-5]\d\d)", re.IGNORECASE)
_SECRET_PATTERNS = (
    re.compile(r"\bBearer\s+[^\s,;]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(api[_ -]?key\s*[=:]\s*)[^\s,;]+"),
)
_SECTION_MARKERS = {
    "<REFERENCES_FILES_START>\n": "reference_files",
    "<SUBMISSION_A_START>\n": "submission_a",
    "<SUBMISSION_B_START>\n": "submission_b",
}
_SECTION_CLOSE_MARKERS = {
    "\n<REFERENCES_FILES_END>\n\n",
    "\n<SUBMISSION_A_END>\n\n",
    "\n<SUBMISSION_B_END>\n\n",
}
_SAFE_RECEIPT_FIELDS = {
    "media_mode",
    "render_dpi",
    "serialized_request_bytes",
    "max_serialized_request_bytes",
    "largest_image_base64_bytes",
    "total_image_base64_bytes",
    "max_image_base64_bytes",
    "max_total_image_base64_bytes",
    "video_file_count",
    "max_video_files",
    "loss_markers",
    "reasons",
}


class JudgeTelemetrySink:
    """Best-effort, process-local JSONL writer.

    Telemetry must never change a judgement. Directory creation and writes are
    therefore swallowed after a warning if the configured destination becomes
    unavailable.
    """

    def __init__(self, output_dir: str | Path | None) -> None:
        self._lock = threading.Lock()
        self._warned = False
        self.path: Path | None = None
        if output_dir:
            directory = Path(output_dir).expanduser()
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.path = directory / f"judge-events.{os.getpid()}.jsonl"
            except OSError as exc:
                self._warn_once(exc)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def _warn_once(self, error: Exception) -> None:
        if not self._warned:
            LOGGER.warning("GDPVal judge telemetry is unavailable: %s", error)
            self._warned = True

    def emit(self, event_type: str, **payload: Any) -> None:
        if self.path is None:
            return
        event = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        try:
            line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except (OSError, TypeError, ValueError) as exc:
            self._warn_once(exc)


def _redact_message(value: object, limit: int = 1000) -> str:
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]" if match.lastindex else "[REDACTED]", text)
    return text[:limit]


def _status_code(error: Exception) -> int | None:
    for candidate in (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            pass
    match = _STATUS_RE.search(str(error))
    return int(match.group("status")) if match else None


def _provider_error_fields(error: Exception) -> tuple[str | None, str | None]:
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        return None, None
    detail = body.get("error", body)
    if not isinstance(detail, Mapping):
        return None, None
    code = detail.get("code")
    provider_type = detail.get("type") or detail.get("status")
    return str(code)[:120] if code is not None else None, str(provider_type)[
        :120
    ] if provider_type is not None else None


def classify_judge_error(error: Exception, *, retryable: bool) -> dict[str, Any]:
    """Return a bounded, credential-free error description."""

    status = _status_code(error)
    lower = str(error).lower()
    error_name = error.__class__.__name__.lower()
    if "transportineligible" in error_name:
        kind = "transport_ineligible"
    elif "all " in lower and "responses were invalid" in lower:
        kind = "invalid_response"
    elif "error getting file" in lower:
        kind = "file_processing"
    elif status == 429 or "rate limit" in lower or "resource_exhausted" in lower:
        kind = "http_429_rate_limit"
    elif status == 413 or "request entity too large" in lower or "request size is too large" in lower:
        kind = "request_too_large"
    elif "timeout" in error.__class__.__name__.lower() or "timed out" in lower or "timeout" in lower:
        kind = "timeout"
    elif status is not None and status >= 500:
        kind = "http_5xx"
    elif status in {401, 403} or "authentication" in lower or "invalid api key" in lower:
        kind = "authentication"
    elif "quota" in lower:
        kind = "quota_exhausted"
    elif "connection" in error.__class__.__name__.lower() or "connection" in lower:
        kind = "connection"
    elif status is not None:
        kind = f"http_{status}"
    else:
        kind = "unknown"

    retry_after: str | None = None
    headers = getattr(getattr(error, "response", None), "headers", None)
    if headers is not None:
        retry_after = headers.get("retry-after")
    provider_code, provider_type = _provider_error_fields(error)
    result: dict[str, Any] = {
        "kind": kind,
        "exception_type": error.__class__.__name__,
        "retryable": retryable,
        "message": _redact_message(error),
    }
    if status is not None:
        result["http_status"] = status
    if retry_after is not None:
        result["retry_after"] = str(retry_after)[:120]
    if provider_code is not None:
        result["provider_code"] = provider_code
    if provider_type is not None:
        result["provider_type"] = provider_type
    return result


def _attachment_info(block: Mapping[str, Any]) -> tuple[str | None, int, int]:
    block_type = block.get("type")
    if block_type in {"image_url", "video_url"}:
        value = block.get(str(block_type), {})
        url = value.get("url", "") if isinstance(value, Mapping) else ""
        if not isinstance(url, str) or not url.startswith("data:"):
            return None, 0, 0
        comma = url.find(",")
        if comma < 0:
            return None, 0, 0
        header = url[5:comma]
        mime = header.split(";", 1)[0] or None
        encoded_chars = len(url) - comma - 1
        padding = int(url.endswith("=")) + int(url.endswith("=="))
        return mime, max(0, (encoded_chars // 4) * 3 - padding), encoded_chars
    if block_type == "input_audio":
        value = block.get("input_audio", {})
        if not isinstance(value, Mapping):
            return None, 0, 0
        data = value.get("data", "")
        if not isinstance(data, str):
            return None, 0, 0
        encoded_chars = len(data)
        padding = int(data.endswith("=")) + int(data.endswith("=="))
        audio_format = str(value.get("format") or "unknown").lower()
        return f"audio/{audio_format}", max(0, (encoded_chars // 4) * 3 - padding), encoded_chars
    return None, 0, 0


def _extension(name: str) -> str:
    path = name.split("!/", 1)[-1]
    suffix = PurePosixPath(path).suffix.lower().lstrip(".")
    return suffix or "none"


def _safe_receipt(receipt: Mapping[str, Any] | None) -> dict[str, Any]:
    if not receipt:
        return {}
    return {key: receipt[key] for key in _SAFE_RECEIPT_FIELDS if key in receipt}


def profile_judge_request(
    messages: Sequence[Mapping[str, Any]],
    *,
    receipt: Mapping[str, Any] | None = None,
    include_files: bool = True,
) -> dict[str, Any]:
    """Describe the transmitted request without copying attachment payloads."""

    section: str | None = None
    current_name: str | None = None
    files: dict[tuple[str, str], dict[str, Any]] = {}
    block_types: Counter[str] = Counter()
    mime_types: Counter[str] = Counter()
    section_file_counts: Counter[str] = Counter()
    raw_attachment_bytes = 0
    encoded_attachment_chars = 0
    attachment_count = 0
    text_chars = 0

    def _file(name: str) -> dict[str, Any]:
        key = (section or "unknown", name)
        if key not in files:
            files[key] = {
                "section": key[0],
                "name": name[:500],
                "extension": _extension(name),
                "attachment_count": 0,
                "attachment_raw_bytes": 0,
                "attachment_encoded_chars": 0,
                "text_chars": 0,
                "content_types": [],
                "mime_types": [],
                "transport_notes": [],
            }
            section_file_counts[key[0]] += 1
        return files[key]

    for message in messages:
        content = message.get("content") or []
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for block in content:
            if not isinstance(block, Mapping):
                continue
            block_type = str(block.get("type") or "unknown")
            block_types[block_type] += 1
            text = str(block.get("text", "")) if block_type == "text" else ""
            if text in _SECTION_MARKERS:
                section = _SECTION_MARKERS[text]
                current_name = None
                continue
            if text in _SECTION_CLOSE_MARKERS:
                section = None
                current_name = None
                continue
            label = _LABEL_RE.match(text)
            structured = _STRUCTURED_LABEL_RE.match(text)
            if section and (label or structured):
                match = label or structured
                assert match is not None
                current_name = match.group("name")
                item = _file(current_name)
                if structured:
                    body = text[structured.end() :]
                    item["text_chars"] += len(body)
                    text_chars += len(body)
                continue

            mime, raw_bytes, encoded_chars = _attachment_info(block)
            if encoded_chars:
                attachment_count += 1
                raw_attachment_bytes += raw_bytes
                encoded_attachment_chars += encoded_chars
                if mime:
                    mime_types[mime] += 1
                if current_name:
                    item = _file(current_name)
                    item["attachment_count"] += 1
                    item["attachment_raw_bytes"] += raw_bytes
                    item["attachment_encoded_chars"] += encoded_chars
                    if block_type not in item["content_types"]:
                        item["content_types"].append(block_type)
                    if mime and mime not in item["mime_types"]:
                        item["mime_types"].append(mime)
            elif block_type == "text":
                text_chars += len(text)
                if current_name and section:
                    item = _file(current_name)
                    item["text_chars"] += len(text)
                    lowered = text.lower()
                    if any(marker in lowered for marker in ("omitted", "not included", "unavailable", "not readable")):
                        item["transport_notes"].append(text.splitlines()[0][:240])

    extension_counts = Counter(item["extension"] for item in files.values())
    result: dict[str, Any] = {
        **_safe_receipt(receipt),
        "message_count": len(messages),
        "content_block_counts": dict(sorted(block_types.items())),
        "file_count": len(files),
        "section_file_counts": dict(sorted(section_file_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "mime_type_counts": dict(sorted(mime_types.items())),
        "attachment_count": attachment_count,
        "raw_attachment_bytes": raw_attachment_bytes,
        "encoded_attachment_chars": encoded_attachment_chars,
        "text_chars": text_chars,
    }
    if include_files:
        result["files"] = list(files.values())
    return result


def compact_request_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in profile.items() if key != "files"}


def _iter_event_files(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(path.glob(EVENT_FILE_GLOB)))
        elif path.is_file():
            files.append(path)
    return files


def summarize_events(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Aggregate judge event logs into operational failure statistics."""

    event_files = _iter_event_files(paths)
    event_counts: Counter[str] = Counter()
    failure_kinds: Counter[str] = Counter()
    http_statuses: Counter[str] = Counter()
    judge_event_counts: Counter[str] = Counter()
    failure_judges: Counter[str] = Counter()
    preflight_reasons: Counter[str] = Counter()
    matchup_failure_kinds: Counter[str] = Counter()
    failed_extensions: Counter[str] = Counter()
    size_buckets: Counter[str] = Counter()
    observed_tasks: set[str] = set()
    affected_tasks: set[str] = set()
    completed = recovered = terminal_failures = 0

    for path in event_files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Skipping malformed telemetry event %s:%d: %s", path, line_number, exc)
                    continue
                event_type = str(event.get("event") or "unknown")
                event_counts[event_type] += 1
                if event.get("task_id"):
                    observed_tasks.add(str(event["task_id"]))
                if event.get("judge"):
                    judge_event_counts[str(event["judge"])] += 1
                if event_type == "judge_attempt_failed":
                    if event.get("task_id"):
                        affected_tasks.add(str(event["task_id"]))
                    if event.get("judge"):
                        failure_judges[str(event["judge"])] += 1
                    error = event.get("error") or {}
                    failure_kinds[str(error.get("kind") or "unknown")] += 1
                    if error.get("http_status") is not None:
                        http_statuses[str(error["http_status"])] += 1
                    request = event.get("request") or {}
                    size = int(request.get("serialized_request_bytes") or 0)
                    mib = size / (1024 * 1024)
                    bucket = (
                        "<50 MiB"
                        if mib < 50
                        else "50-100 MiB"
                        if mib < 100
                        else "100-250 MiB"
                        if mib < 250
                        else ">=250 MiB"
                    )
                    size_buckets[bucket] += 1
                    for extension, count in (request.get("extension_counts") or {}).items():
                        failed_extensions[str(extension)] += int(count)
                elif event_type == "judge_preflight_excluded":
                    if event.get("task_id"):
                        affected_tasks.add(str(event["task_id"]))
                    if event.get("judge"):
                        failure_judges[str(event["judge"])] += 1
                    for reason in (event.get("request") or {}).get("reasons") or []:
                        preflight_reasons[str(reason)] += 1
                elif event_type == "judge_request_completed":
                    completed += 1
                    outcome = event.get("outcome")
                    recovered += int(outcome == "recovered")
                    terminal_failures += int(outcome == "failed")
                elif event_type == "judge_response_invalid" and event.get("task_id"):
                    affected_tasks.add(str(event["task_id"]))
                elif event_type == "judge_matchup_failed":
                    if event.get("task_id"):
                        affected_tasks.add(str(event["task_id"]))
                    error = event.get("error") or {}
                    matchup_failure_kinds[str(error.get("kind") or "unknown")] += 1

    return {
        "files": [str(path) for path in event_files],
        "event_counts": dict(event_counts.most_common()),
        "completed_requests": completed,
        "terminal_failures": terminal_failures,
        "terminal_failure_rate": terminal_failures / completed if completed else None,
        "recovered_requests": recovered,
        "observed_task_count": len(observed_tasks),
        "affected_task_count": len(affected_tasks),
        "failure_kinds": dict(failure_kinds.most_common()),
        "http_statuses": dict(http_statuses.most_common()),
        "judge_event_counts": dict(judge_event_counts.most_common()),
        "failure_judges": dict(failure_judges.most_common()),
        "preflight_reasons": dict(preflight_reasons.most_common()),
        "matchup_failure_kinds": dict(matchup_failure_kinds.most_common()),
        "failure_request_size_buckets": dict(size_buckets.most_common()),
        "extensions_on_failed_attempts": dict(failed_extensions.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GDPVal judge telemetry JSONL files")
    parser.add_argument("paths", nargs="+", help="Telemetry JSONL file(s) or directories")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    args = parser.parse_args()
    print(json.dumps(summarize_events(args.paths), indent=args.indent, sort_keys=True))


if __name__ == "__main__":
    main()
