# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Request validation errors: bounded, escaped logs of what was wrong, and the 422 response FastAPI returns."""

import json
import logging
from typing import Any

from fastapi import Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError


logger = logging.getLogger(__name__)


# Bound both the raw request prefix and its escaped representation to 4 KiB.
_VALIDATION_ERROR_LOG_BODY_CHARS = 4096


_VALIDATION_ERROR_LOG_MAX_ERRORS = 20


_VALIDATION_ERROR_LOG_FIELD_CHARS = 256


_VALIDATION_ERROR_LOG_LOC_ITEMS = 8


# Bound the rendered error summaries in the log message to 4 KiB.
_VALIDATION_ERROR_LOG_ERRORS_CHARS = 4096


def _escaped_log_text(value: str, max_chars: int) -> tuple[str, bool]:
    """Return bounded, control-safe text with a visible marker when truncated."""
    raw_prefix = value[:max_chars]
    rendered = json.dumps(raw_prefix, ensure_ascii=True)[1:-1]
    truncated = len(value) > len(raw_prefix) or len(rendered) > max_chars
    suffix = "...[truncated]" if truncated else ""
    # The marker shares the existing budget; preserve complete JSON escapes.
    content_limit = max_chars - len(suffix)
    safe_end = 0
    index = 0
    while index < len(rendered) and index < content_limit:
        escape_chars = 1
        if rendered[index] == "\\":
            escape_chars = 6 if index + 1 < len(rendered) and rendered[index + 1] == "u" else 2
        if index + escape_chars > content_limit:
            break
        index += escape_chars
        safe_end = index

    return rendered[:safe_end] + suffix, truncated


def _escaped_log_prefix(value: str, max_chars: int) -> tuple[str, bool]:
    """Return a quoted, control-safe prefix bounded by ``max_chars``."""
    escaped, truncated = _escaped_log_text(value, max_chars - 2)
    return f'"{escaped}"', truncated


def _bounded_validation_error_value(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        return _escaped_log_text(value, _VALIDATION_ERROR_LOG_FIELD_CHARS)
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    return f"<{type(value).__name__}>", True


def _validation_error_summaries(errors: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    summaries = []
    truncated = len(errors) > _VALIDATION_ERROR_LOG_MAX_ERRORS
    for error in errors[:_VALIDATION_ERROR_LOG_MAX_ERRORS]:
        error_type, type_truncated = _bounded_validation_error_value(error.get("type"))
        message, message_truncated = _bounded_validation_error_value(error.get("msg"))
        location = error.get("loc")
        if isinstance(location, (list, tuple)):
            location_items = []
            location_truncated = len(location) > _VALIDATION_ERROR_LOG_LOC_ITEMS
            for item in location[:_VALIDATION_ERROR_LOG_LOC_ITEMS]:
                bounded_item, item_truncated = _bounded_validation_error_value(item)
                location_items.append(bounded_item)
                location_truncated = location_truncated or item_truncated
        else:
            bounded_location, location_truncated = _bounded_validation_error_value(location)
            location_items = [bounded_location]

        summaries.append({"type": error_type, "loc": location_items, "msg": message})
        truncated = truncated or type_truncated or location_truncated or message_truncated

    return summaries, truncated


def _validation_errors_log_text(error_summaries: list[dict[str, Any]]) -> tuple[str, bool]:
    """Render the error summaries for the log message, bounded by ``_VALIDATION_ERROR_LOG_ERRORS_CHARS``."""
    # Custom validator messages can echo credentials from headers, queries, paths, or cookies.
    # Keep their type and location visible without copying their messages into the default log.
    message_summaries = [
        summary if summary["loc"] and summary["loc"][0] == "body" else {key: summary[key] for key in ("type", "loc")}
        for summary in error_summaries
    ]
    rendered = json.dumps(message_summaries, ensure_ascii=True, separators=(",", ":"))
    if len(rendered) <= _VALIDATION_ERROR_LOG_ERRORS_CHARS:
        return rendered, False
    suffix = "...[truncated]"
    return rendered[: _VALIDATION_ERROR_LOG_ERRORS_CHARS - len(suffix)] + suffix, True


async def _log_validation_exception(request: Request, exc: RequestValidationError) -> None:
    errors = exc.errors()
    error_summaries, errors_truncated = _validation_error_summaries(errors)
    # The summaries are also attached as ``extra``, but the default formatter drops extras,
    # so render them into the message too: a 422 log line should say which field failed.
    errors_text, errors_text_truncated = _validation_errors_log_text(error_summaries)
    errors_log_truncated = errors_truncated or errors_text_truncated
    errors_suffix = " ...[truncated]" if errors_log_truncated else ""
    extra = {
        "validation_error_count": len(errors),
        "validation_errors": error_summaries,
        "validation_errors_truncated": errors_truncated,
    }

    has_body_error = any(
        isinstance(error.get("loc"), (list, tuple)) and error["loc"] and error["loc"][0] == "body" for error in errors
    )
    if not has_body_error:
        logger.warning(
            "Request validation failed; validation_error_count=%d validation_errors=%s "
            "validation_errors_truncated=%s%s",
            len(errors),
            errors_text,
            errors_log_truncated,
            errors_suffix,
            extra=extra,
        )
        return

    try:
        body = await request.body()
    except Exception:
        logger.warning(
            "Request validation failed; request body unavailable; "
            "validation_error_count=%d validation_errors=%s validation_errors_truncated=%s%s",
            len(errors),
            errors_text,
            errors_log_truncated,
            errors_suffix,
            extra=extra,
        )
        return

    raw_prefix = body[:_VALIDATION_ERROR_LOG_BODY_CHARS]
    escaped_prefix, prefix_truncated = _escaped_log_prefix(
        raw_prefix.decode("utf-8", errors="replace"), _VALIDATION_ERROR_LOG_BODY_CHARS
    )
    body_truncated = len(body) > len(raw_prefix) or prefix_truncated
    extra.update(
        {
            "request_body_size_bytes": len(body),
            "request_body_prefix": escaped_prefix,
            "request_body_truncated": body_truncated,
        }
    )
    logger.warning(
        "Request validation failed; request_body_size_bytes=%d request_body_truncated=%s request_body_prefix=%s "
        "validation_error_count=%d validation_errors=%s validation_errors_truncated=%s%s",
        len(body),
        body_truncated,
        escaped_prefix,
        len(errors),
        errors_text,
        errors_log_truncated,
        errors_suffix,
        extra=extra,
    )


async def _validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
    try:
        await _log_validation_exception(request, exc)
    except Exception:
        # Diagnostics must not alter FastAPI's response contract.
        pass
    return await request_validation_exception_handler(request, exc)
