# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution health independent of verifier reward and optional observability."""

import re
from typing import Any


def execution_health(
    *,
    return_code: int | None,
    error_type: str | None,
    exception_type: str | None,
    elapsed_s: float,
    budget_s: float,
    finished: bool,
    export: dict[str, Any],
    control_error: str | None = None,
) -> dict[str, Any]:
    assistants = [
        m.get("info", {}) for m in export.get("messages", []) if m.get("info", {}).get("role") == "assistant"
    ]
    last = assistants[-1] if assistants else {}
    error = last.get("error") or {}
    data = error.get("data") or {}
    status_code = data.get("statusCode")
    message = str(data.get("message", ""))
    context_limit = error.get("name") == "ContextOverflowError" or (
        status_code == 400 and re.search(r"maximum context length|context window|too many tokens", message, re.I)
    )
    killed = return_code == -1 and (control_error or "").strip() == "signal: killed"
    timeout = "timeout" in str(error_type or "").lower() or return_code == 124 or killed
    budget_expired = exception_type is None and budget_s > 0 and elapsed_s >= budget_s and timeout
    reason = None
    outcome = "completed"
    if error and not context_limit:
        outcome = "infrastructure_error"
        transport = re.search(
            r"ECONNREFUSED|ECONNRESET|ETIMEDOUT|ENOTFOUND|EAI_AGAIN|connection refused|"
            r"connection reset|unable to connect|cannot connect to API|fetch failed|socket hang up|request timed out",
            message,
            re.I,
        )
        if transport or (isinstance(status_code, int) and (status_code >= 500 or status_code in (408, 429))):
            reason = "model_endpoint_error"
        else:
            reason = "model_api_error"
    elif budget_expired:
        outcome = "agent_timeout"
    elif exception_type or error_type or return_code != 0:
        outcome, reason = "infrastructure_error", "agent_execution_error"
    elif context_limit or last.get("finish") == "length":
        outcome = "context_limit"
    elif not finished:
        outcome, reason = "infrastructure_error", "agent_did_not_finish"
    elif not assistants or last.get("finish") != "stop":
        outcome, reason = "infrastructure_error", "agent_completion_evidence_missing"
    return {
        "schema_version": 1,
        "outcome": outcome,
        "retryable": outcome == "infrastructure_error",
        "reason": reason,
        "return_code": return_code,
        "exec_error_type": error_type,
        "exception_type": exception_type,
        "elapsed_s": elapsed_s,
        "budget_s": budget_s,
        "budget_termination_inferred_from_signal": bool(budget_expired and killed),
        "finished_marker": finished,
        "export_found": bool(export),
        "last_model_finish": last.get("finish"),
        "terminal_model_error": {"name": error.get("name"), "http_status": status_code} if error else None,
    }
