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
}
# Reference raster paths are the upstream author's local files; the raster itself
# is identified by the preceding get_raster_path step's `rasterdataset`.
RASTER_PATH_KEYS = {"raster_path", "raster1_path", "raster2_path"}
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
# Bounds scoring time for adversarial trajectories with many interchangeable calls.
# The search order is deterministic, so a truncated search still scores reproducibly.
MAX_SEARCH_NODES = 20_000


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


Bindings = dict[str, tuple[str, int]]


def _consumed_handles(arguments: dict[str, Any]) -> list[str]:
    handles = []
    for key in HANDLE_KEYS:
        value = arguments.get(key)
        handles.extend(item for item in (value if isinstance(value, list) else [value]) if isinstance(item, str))
    return handles


def is_well_formed(steps: list[dict[str, Any]]) -> bool:
    """Every handle a reference step consumes must be produced by an earlier step."""
    produced: set[str] = set()
    for step in steps:
        arguments = step.get("arguments")
        if not isinstance(arguments, dict) or not isinstance(step.get("function_name"), str):
            return False
        if any(handle not in produced for handle in _consumed_handles(arguments)):
            return False
        produced.update(arguments[key] for key in OUTPUT_KEYS if isinstance(arguments.get(key), str))
    return True


def _step_match(
    candidate: tuple[str, dict[str, Any] | None],
    call_index: int,
    reference: dict[str, Any],
    bindings: Bindings,
) -> tuple[bool, bool, Bindings]:
    """Return (essential_match, partial_difference, bindings after this step).

    `bindings` maps a reference handle to the candidate handle and the index of the
    candidate call that produced it; a consumer must come after its producers.
    """
    name, candidate_args = candidate
    if name != reference["function_name"] or candidate_args is None:
        return False, False, bindings
    reference_args = reference.get("arguments") or {}
    if not isinstance(reference_args, dict):
        return False, False, bindings
    partial = False
    next_bindings = dict(bindings)

    def consumes(expected: Any, actual: Any) -> bool:
        bound = bindings.get(expected)
        return bound is not None and bound[0] == actual and bound[1] < call_index

    for key, expected in reference_args.items():
        actual = candidate_args.get(key)
        if key in OUTPUT_KEYS:
            if isinstance(expected, str) and isinstance(actual, str):
                next_bindings[expected] = (actual, call_index)
            continue
        if key in HANDLE_KEYS:
            if isinstance(expected, str):
                if not consumes(expected, actual):
                    return False, partial, bindings
            elif isinstance(expected, list):
                if not isinstance(actual, list) or len(actual) != len(expected):
                    return False, partial, bindings
                if any(
                    not consumes(value, item) if isinstance(value, str) else _norm(value) != _norm(item)
                    for value, item in zip(expected, actual)
                ):
                    return False, partial, bindings
            elif _norm(actual) != _norm(expected):
                return False, partial, bindings
            continue
        if key in NONESSENTIAL_KEYS or key in RASTER_PATH_KEYS:
            continue
        if key in CATALOG_KEYS:
            if actual != expected:
                return False, partial, bindings
            continue
        if _norm(actual) == _norm(expected):
            continue
        if key in PARTIAL_KEYS:
            partial = True
            continue
        return False, partial, bindings
    return True, partial, next_bindings


def _match_steps(
    calls: list[tuple[str, dict[str, Any] | None]], steps: list[dict[str, Any]]
) -> tuple[int, bool, bool]:
    """Return (matched step count, all matched steps exact, search exhausted).

    Reference steps are visited in reference order, which puts producers before
    consumers; candidate calls may be matched in any order that keeps each consumer
    after its producers, so independent calls commute.
    """
    best = (0, False)
    perfect = (len(steps), True)
    names = {name for name, arguments in calls if arguments is not None}
    # Upper bound on how many of steps[index:] could still match.
    possible = [0] * (len(steps) + 1)
    # Reference handles consumed by steps[index:]; other bindings no longer matter.
    consumed_later: list[frozenset[str]] = [frozenset()] * (len(steps) + 1)
    for index in reversed(range(len(steps))):
        step = steps[index]
        arguments = step.get("arguments") if isinstance(step.get("arguments"), dict) else {}
        possible[index] = possible[index + 1] + (step.get("function_name") in names)
        consumed_later[index] = consumed_later[index + 1] | frozenset(_consumed_handles(arguments))
    seen: set[tuple] = set()
    budget = MAX_SEARCH_NODES

    def visit(index: int, unused: frozenset[int], bindings: Bindings, matched: int, exact: bool) -> None:
        nonlocal best, budget
        best = max(best, (matched, exact))
        if best == perfect or budget <= 0 or (matched + possible[index], exact) <= best:
            return
        relevant = tuple(sorted(item for item in bindings.items() if item[0] in consumed_later[index]))
        state = (index, unused, relevant, exact)
        if state in seen:
            return
        seen.add(state)
        budget -= 1
        reference = steps[index]
        options = []
        for call_index in sorted(unused):
            ok, is_partial, next_bindings = _step_match(calls[call_index], call_index, reference, bindings)
            if ok:
                options.append((is_partial, call_index, next_bindings))
        # Exact candidates first, but keep partial candidates as fallbacks.
        for is_partial, call_index, next_bindings in sorted(options, key=lambda item: item[:2]):
            visit(index + 1, unused - {call_index}, next_bindings, matched + 1, exact and not is_partial)
        visit(index + 1, unused, bindings, matched, exact)

    visit(0, frozenset(range(len(calls))), {}, 0, True)
    return best[0], best[1], budget > 0


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
        score = 2 if calls and calls[-1][0] == "reject_task" and calls[-1][1] is not None else 0
        return score, {"reference_tools": reference_names, "matched_steps": int(score == 2)}
    if "reject_task" in candidate_names:
        return 0, {"reference_tools": reference_names, "matched_steps": 0}

    well_formed = is_well_formed(steps)
    matched, exact, exhaustive = _match_steps(calls, steps)

    overlap = sum((Counter(reference_names) & Counter(candidate_names)).values())
    coverage = matched / len(steps)
    tool_coverage = overlap / len(steps)
    if coverage == 1 and well_formed:
        score = 2 if exact else 1
    elif max(coverage, tool_coverage) >= 0.6:
        score = 1
    else:
        score = 0
    return score, {
        "reference_tools": reference_names,
        "matched_steps": matched,
        "reference_steps": len(steps),
        "reference_well_formed": well_formed,
        "search_exhaustive": exhaustive,
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
