# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Shared utilities for tool action implementations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


def _fill_template(value: Any, params: Dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {k: _fill_template(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_template(v, params) for v in value]
    if isinstance(value, str):
        if value.startswith("sample_"):
            key = value[len("sample_") :]
            if key in params:
                return params[key]
        if value == "sample_id":
            for key in ("id", "record_id", "task_id"):
                if key in params:
                    return params[key]
        return value
    return value


def _build_response(
    template: Optional[Dict[str, Any]], results: List[Dict[str, Any]], params: Dict[str, Any]
) -> Dict[str, Any]:
    if template:
        response = deepcopy(template)
    else:
        response = {"success": True, "results": []}
    response["success"] = True
    raw_results = response.get("results")
    template_results: List[Dict[str, Any]] | None = (
        raw_results if isinstance(raw_results, list) else None
    )
    if results:
        merged_results: List[Dict[str, Any]] = []
        for result in results:
            base = template_results[0] if template_results else {}
            filled_base = _fill_template(base, params)
            merged = {**filled_base, **result}
            merged_results.append(merged)
        response["results"] = merged_results
    else:
        response["results"] = []
    final_results = response.get("results")
    response["count"] = len(final_results) if isinstance(final_results, list) else 0
    return response
