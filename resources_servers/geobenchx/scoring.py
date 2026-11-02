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

"""Deterministic GeoBenchX reference matcher for regression and training rewards.

The paper used a three-model LLM judge. A deterministic matcher is the default
here so release-to-release regression scores are reproducible. It follows the
paper's 0/1/2 taxonomy and can coexist with a future panel-judge verifier.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any


OUTPUT_KEYS = {
    "output_dataframe_name",
    "output_geodataframe_name",
    "output_variable_name",
    "output_column_name",
    "output_filename",
    "output_folder",
}
HANDLE_KEYS = {
    "dataframe_name",
    "dataframe1_name",
    "dataframe2_name",
    "geodataframe_name",
    "features_geodataframe_name",
    "reference_geodataframe_name",
    "points_geodataframe_name",
    "geodataframe_names",
    "raster_path",
}
NONESSENTIAL_KEYS = {
    "legendtext",
    "title",
    "plot_title",
    "plot_result",
    "cleanup_files",
    "width",
    "height",
    "zoom_level",
    "radius",
    "map_style",
    "basemap_style",
    "add_legend",
    "layer_styles",
    "figsize",
    "add_colorbar",
    "plot_background",
}
PARTIAL_KEYS = {"colormap", "nodataval", "spatial_predicates"}
CATALOG_KEYS = {"dataset", "geodataset", "rasterdataset"}
MAX_TOOL_CALLS = 25


def parse_calls(output: list[Any]) -> list[tuple[str, dict[str, Any] | None]]:
    """Keep malformed calls visible: the agent attempted them, but they cannot match."""
    calls = []
    for item in output:
        payload = item.model_dump() if hasattr(item, "model_dump") else item
        if not isinstance(payload, dict) or payload.get("type") != "function_call":
            continue
        arguments = payload.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        calls.append((str(payload.get("name", "")), arguments if isinstance(arguments, dict) else None))
    return calls


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return _norm(json.loads(stripped))
        except (json.JSONDecodeError, TypeError):
            return " ".join(stripped.lower().split())
    if isinstance(value, dict):
        return {str(key).lower(): _norm(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return sorted((_norm(item) for item in value), key=str)
    if isinstance(value, float):
        return round(value, 6)
    return value


def _step_match(
    candidate: tuple[str, dict[str, Any] | None],
    reference: dict[str, Any],
    bindings: dict[str, str],
) -> tuple[bool, bool, dict[str, str]]:
    """Return (essential_match, partial_difference)."""
    name, candidate_args = candidate
    if name != reference["function_name"] or candidate_args is None:
        return False, False, bindings
    reference_args = reference.get("arguments") or {}
    partial = False
    next_bindings = dict(bindings)
    for key, expected in reference_args.items():
        actual = candidate_args.get(key)
        if key in OUTPUT_KEYS:
            if isinstance(expected, str) and isinstance(actual, str):
                bound = next_bindings.get(expected)
                if bound is not None and bound != actual:
                    return False, partial, bindings
                next_bindings[expected] = actual
            continue
        if key in HANDLE_KEYS:
            if isinstance(expected, str):
                if expected not in bindings or actual != bindings[expected]:
                    return False, partial, bindings
            elif isinstance(expected, list):
                if not isinstance(actual, list) or len(actual) != len(expected):
                    return False, partial, bindings
                if any(
                    isinstance(value, str) and (value not in bindings or candidate != bindings[value])
                    for value, candidate in zip(expected, actual)
                ):
                    return False, partial, bindings
            elif _norm(actual) != _norm(expected):
                return False, partial, bindings
            continue
        if key in NONESSENTIAL_KEYS:
            continue
        if key in CATALOG_KEYS and actual != expected:
            return False, partial, bindings
        if _norm(actual) == _norm(expected):
            continue
        if key in PARTIAL_KEYS:
            partial = True
            continue
        return False, partial, bindings
    return True, partial, next_bindings


def _match_steps(
    calls: list[tuple[str, dict[str, Any] | None]], steps: list[dict[str, Any]]
) -> tuple[int, bool]:
    """Backtrack over ready reference steps so independent calls may commute."""
    best = (0, False)

    def visit(remaining: set[int], unused: set[int], bindings: dict[str, str], matched: int, partial: bool) -> None:
        nonlocal best
        best = max(best, (matched, not partial))
        for ref_index in remaining:
            reference = steps[ref_index]
            # A consumer cannot be matched until its reference handles are bound.
            args = reference.get("arguments") or {}
            if any(isinstance(args.get(key), str) and args[key] not in bindings for key in HANDLE_KEYS):
                continue
            options = []
            for call_index in unused:
                ok, is_partial, next_bindings = _step_match(calls[call_index], reference, bindings)
                if ok:
                    options.append((call_index, is_partial, next_bindings))
            # Prefer exact candidates, but keep partial candidates as fallbacks.
            for call_index, is_partial, next_bindings in sorted(options, key=lambda item: item[1]):
                visit(
                    remaining - {ref_index},
                    unused - {call_index},
                    next_bindings,
                    matched + 1,
                    partial or is_partial,
                )

    visit(set(range(len(steps))), set(range(len(calls))), {}, 0, False)
    return best[0], not best[1]


def _score_solution(
    calls: list[tuple[str, dict[str, Any] | None]], reference: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    steps = reference.get("steps") or []
    reference_names = [step["function_name"] for step in steps]
    candidate_names = [name for name, _ in calls]
    if not steps:
        score = 2 if not calls else 0
        return score, {"reference_tools": [], "matched_steps": 0}
    if reference_names == ["reject_task"]:
        # The paper's judge: a candidate that stops with reject_task fully matches a
        # reject_task reference, whatever it inspected first.
        score = 2 if calls and calls[-1] == ("reject_task", {}) else 0
        return score, {"reference_tools": reference_names, "matched_steps": int(score == 2)}
    if "reject_task" in candidate_names:
        return 0, {"reference_tools": reference_names, "matched_steps": 0}

    matched, exact = _match_steps(calls, steps)

    overlap = sum((Counter(reference_names) & Counter(candidate_names)).values())
    coverage = matched / len(steps)
    tool_coverage = overlap / len(steps)
    if coverage == 1:
        score = 2 if exact else 1
    elif max(coverage, tool_coverage) >= 0.6:
        score = 1
    else:
        score = 0
    return score, {
        "reference_tools": reference_names,
        "matched_steps": matched,
        "reference_steps": len(steps),
        "semantic_coverage": round(coverage, 6),
        "tool_coverage": round(tool_coverage, 6),
        "extra_steps": max(0, len(calls) - len(steps)),
    }


def score_task(calls: list[tuple[str, dict[str, Any] | None]], task: dict[str, Any]) -> dict[str, Any]:
    if len(calls) > MAX_TOOL_CALLS:
        return {
            "reward": 0.0,
            "geobenchx_match": 0,
            "rejected": any(name == "reject_task" for name, _ in calls),
            "candidate_steps": len(calls),
            "reference_steps_mean": 0.0,
            "relative_length_delta": 0.0,
            "matched_reference_index": None,
            "match_detail": {"error": f"tool-call limit exceeded ({len(calls)} > {MAX_TOOL_CALLS})"},
        }
    scored = [_score_solution(calls, solution) for solution in task["reference_solutions"]]
    best_index = max(range(len(scored)), key=lambda index: scored[index][0])
    match, detail = scored[best_index]
    reference_lengths = [len(solution.get("steps") or []) for solution in task["reference_solutions"]]
    mean_reference_length = sum(reference_lengths) / len(reference_lengths)
    candidate_length = len(calls)
    length_delta = (
        (candidate_length - mean_reference_length) / candidate_length if candidate_length else -mean_reference_length
    )
    return {
        "reward": match / 2.0,
        "geobenchx_match": match,
        "rejected": any(name == "reject_task" for name, _ in calls),
        "candidate_steps": candidate_length,
        "reference_steps_mean": round(mean_reference_length, 6),
        "relative_length_delta": round(length_delta, 6),
        "matched_reference_index": best_index,
        "match_detail": detail,
    }
