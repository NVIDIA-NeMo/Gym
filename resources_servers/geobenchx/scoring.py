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


def parse_calls(output: list[Any]) -> list[tuple[str, dict[str, Any]]]:
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
                arguments = {}
        calls.append((str(payload.get("name", "")), arguments if isinstance(arguments, dict) else {}))
    return calls[:25]


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


def _step_match(candidate: tuple[str, dict[str, Any]], reference: dict[str, Any]) -> tuple[bool, bool]:
    """Return (essential_match, partial_difference)."""
    name, candidate_args = candidate
    if name != reference["function_name"]:
        return False, False
    reference_args = reference.get("arguments") or {}
    partial = False
    for key, expected in reference_args.items():
        if key in OUTPUT_KEYS | HANDLE_KEYS | NONESSENTIAL_KEYS:
            continue
        actual = candidate_args.get(key)
        if _norm(actual) == _norm(expected):
            continue
        if key in PARTIAL_KEYS:
            partial = True
            continue
        return False, partial
    return True, partial


def _score_solution(calls: list[tuple[str, dict[str, Any]]], reference: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    steps = reference.get("steps") or []
    reference_names = [step["function_name"] for step in steps]
    candidate_names = [name for name, _ in calls]
    if not steps:
        score = 2 if not calls else 0
        return score, {"reference_tools": [], "matched_steps": 0}
    if reference_names == ["reject_task"]:
        # The paper's judge: a candidate that stops with reject_task fully matches a
        # reject_task reference, whatever it inspected first.
        score = 2 if candidate_names and candidate_names[-1] == "reject_task" else 0
        return score, {"reference_tools": reference_names, "matched_steps": int(score == 2)}
    if "reject_task" in candidate_names:
        return 0, {"reference_tools": reference_names, "matched_steps": 0}

    unmatched = set(range(len(calls)))
    matched = 0
    partial = False
    for step in steps:
        options = [
            (index, _step_match(calls[index], step)) for index in unmatched if calls[index][0] == step["function_name"]
        ]
        exact = next(((index, result) for index, result in options if result[0]), None)
        if exact:
            index, result = exact
            unmatched.remove(index)
            matched += 1
            partial = partial or result[1]

    overlap = sum((Counter(reference_names) & Counter(candidate_names)).values())
    coverage = matched / len(steps)
    tool_coverage = overlap / len(steps)
    if coverage == 1:
        score = 1 if partial else 2
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


def score_task(calls: list[tuple[str, dict[str, Any]]], task: dict[str, Any]) -> dict[str, Any]:
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
