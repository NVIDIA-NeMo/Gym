# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent, data-only next-call comparison; no judge or tool execution."""

import hashlib
import json
import math
from typing import Any


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def strict_arguments(text: str) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_argument_key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite_argument")

    value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("arguments_must_be_object")

    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite_argument")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        if isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def _normalized_scalar(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).lower().strip()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text


def compare_value(expected: Any, actual: Any, profile: str) -> bool:
    if profile == "case_insensitive_scalars":
        return expected == actual or str(expected).lower() == str(actual).lower()
    if profile == "normalized_scalars":
        return _normalized_scalar(expected) == _normalized_scalar(actual)
    raise ValueError("unknown comparison_profile")


def score_native(
    output: list[dict[str, Any]], expected_function: str, expected_params: dict[str, Any], comparison_profile: str
) -> dict[str, Any]:
    """First actual generated call, expected-parameter subset, explicit diagnostics.

    Text describing a call does not count. Malformed JSON and non-object
    arguments fail, even when no parameters are expected. Extra arguments and
    later calls are reported and ignored under this captured comparison profile.
    """
    calls = [item for item in output if item.get("type") == "function_call"]
    result = {
        "reward": 0.0,
        "comparison_profile": comparison_profile,
        "parameter_scope": "expected_subset",
        "predicted_function": None,
        "function_correct": False,
        "params_total": len(expected_params),
        "params_correct": 0,
        "parameter_checks": {},
        "extra_argument_keys": [],
        "missing_expected_keys": [],
        "ignored_call_count": max(0, len(calls) - 1),
        "arguments_valid": False,
    }
    if not calls:
        result["reason"] = "no_generated_function_call"
        return result
    call = calls[0]
    result["predicted_function"] = call.get("name")
    result["function_correct"] = call.get("name") == expected_function
    try:
        arguments = strict_arguments(call.get("arguments"))
    except (ValueError, TypeError, RecursionError):
        result["reason"] = "malformed_function_arguments"
        return result
    result["arguments_valid"] = True
    result["extra_argument_keys"] = sorted(set(arguments) - set(expected_params))
    result["missing_expected_keys"] = sorted(set(expected_params) - set(arguments))
    result["parameter_checks"] = {
        key: compare_value(expected, arguments.get(key), comparison_profile)
        for key, expected in expected_params.items()
    }
    result["params_correct"] = sum(result["parameter_checks"].values())
    result["reward"] = float(result["function_correct"] and all(result["parameter_checks"].values()))
    result["reason"] = "matched_native_reference" if result["reward"] else "native_reference_mismatch"
    return result
