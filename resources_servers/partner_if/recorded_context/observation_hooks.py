# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual executor measurements, separate from model-visible frozen fixtures."""

from time import perf_counter, time


def tool_observation(
    call: dict,
    result: str,
    kind: str,
    state,
    started_at: float,
    started_monotonic: float,
    *,
    conversation_id: str = "parent",
) -> dict:
    failed = kind in {"fixture_gap", "invalid_call", "file_error"}
    exit_code = state.readonly.last_status if kind == "readonly_tool_result" and state.readonly is not None else None
    failed = failed or (exit_code is not None and exit_code != 0)
    return {
        "conversation_id": conversation_id,
        "tool_call_id": call["id"],
        "function": call["function"],
        "kind": kind,
        "result": result,
        "started_at": started_at,
        "completed_at": max(started_at, time()),
        "duration_ms": (perf_counter() - started_monotonic) * 1000,
        "status": "failed" if failed else "completed",
        "error_type": (f"tool_exit_{exit_code}" if exit_code else kind) if failed else None,
    }
