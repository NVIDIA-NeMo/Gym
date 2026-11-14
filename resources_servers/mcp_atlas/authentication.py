# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recognize credential failures only in trusted, failed service responses."""

from __future__ import annotations

import json
import re
from collections.abc import Collection


if __package__:
    from .credentials import SERVER_CREDENTIALS
else:
    from credentials import SERVER_CREDENTIALS


def tool_server_name(tool_name: str, servers: Collection[str]) -> str | None:
    """Attribute a full task tool name; a bare sandbox alias is not a service identity."""
    matches = [server for server in servers if tool_name.startswith(server + "_")]
    return max(matches, key=len) if matches else None


def tool_error_diagnostic(text: str, tool_name: str) -> tuple[str, bool]:
    """Unwrap the pinned bridge's named-tool exception without interpreting successful content."""
    error = text.strip()
    wrapped = False
    for prefix in (f"Failed to call tool '{tool_name}':", "Failed to call tool:", "Failed to list tools:"):
        if error.startswith(prefix):
            error = error[len(prefix) :].lstrip()
            wrapped = True
            break
    inner_prefix = f"500: Tool '{tool_name}' execution failed:"
    if wrapped and error.startswith(inner_prefix):
        error = error[len(inner_prefix) :].lstrip()
    if error.startswith("Error:"):
        error = error[len("Error:") :].lstrip()
        wrapped = True
    return error, wrapped


def is_authentication_failure(
    status: int,
    text: str,
    *,
    servers: Collection[str],
    tool_name: str | None = None,
    served_name: str | None = None,
) -> bool:
    """Classify a failed credentialed service, never an opaque HTTP 200 tool result.

    A tool is attributed by its full task name, while ``served_name`` only unwraps a single-server alias.
    With no tool name the caller must supply an explicit startup-failure diagnostic, not a health payload
    or arbitrary model content. The caller uses an error status for such recognized startup failures.
    """
    if status < 400:
        return False
    if tool_name is not None:
        server = tool_server_name(tool_name, servers)
        contexts = {server} if server else set()
    else:
        contexts = set(servers)
    contexts = {server for server in contexts if SERVER_CREDENTIALS.get(server)}
    if not contexts:
        return False
    if status == 401:
        return True

    diagnostic = text[:2000].strip()
    try:
        body = json.loads(diagnostic)
    except (TypeError, ValueError):
        body = None
    if isinstance(body, dict):
        diagnostic = next(
            (body[key] for key in ("detail", "message", "error") if isinstance(body.get(key), str)), diagnostic
        )
    # Select the actual wrapper once. Trying both names independently can reinterpret an unmatched alias.
    wrapper_name = served_name or tool_name or ""
    if tool_name and diagnostic.startswith(f"Failed to call tool '{tool_name}':"):
        wrapper_name = tool_name
    diagnostic, _ = tool_error_diagnostic(diagnostic, wrapper_name)
    match = re.match(
        r"^(?:(?:HTTP(?:/\d(?:\.\d)?)?\s+|HTTPError:\s*|ClientResponseError:\s*)?)(401|403)\b[\s:,-]*(.*)",
        diagnostic,
        re.IGNORECASE | re.DOTALL,
    )
    service_status = int(match[1]) if match else status
    detail = match[2] if match else diagnostic
    # Forbidden is an HTTP/access outcome, not proof that credentials or configured scopes are invalid.
    detail = re.sub(r"^Forbidden\b(?:[\s:,-]+|$)", "", detail, count=1, flags=re.IGNORECASE)
    # GitHub uses 403 for both primary and secondary rate limits. Match its canonical diagnostics,
    # not arbitrary occurrences of "rate", "quota", or words inside model arguments.
    if (
        "github" in contexts
        and service_status == 403
        and re.match(
            r"^(?:API rate limit exceeded\b|You have exceeded a secondary rate limit\b)", detail, re.IGNORECASE
        )
    ):
        return False
    if service_status == 401:
        return True
    if "slack" in contexts and re.match(
        r"^(?:invalid_auth|not_authed|token_expired|token_revoked|account_inactive|missing_scope)\b", detail
    ):
        return True
    return (
        re.match(
            r"^(?:unauthori[sz]ed\b|authentication failed\b|bad credentials\b|"
            r"insufficient(?: authentication| OAuth)? scopes?\b|insufficient_scope\b|"
            r"invalid credentials\b|(?:invalid|missing) API[ _-]?key(?:[.!:]|$)|"
            r"invalid (?:access|auth) token(?:[.!:]|$))",
            detail,
            re.IGNORECASE,
        )
        is not None
    )
