# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select the upstream prompt and scorer for an explicit task rubric."""

from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from resources_servers.vision2code import generic_rubric, rubrics


RubricName = Literal[
    "ChartQA",
    "dvqa",
    "figureqa",
    "ChemVQA-2K",
    "ChemVQA-2K-spacing-v1",
    "DocVQA",
    "EEE-Bench",
    "GEOQA_8K_R1V",
    "Geoperception",
    "geometry3k",
    "Graph-Algorithms",
    "GraphVQA-Swift",
    "matplotlib",
    "OlympiadBench",
    "Physics",
    "spatialvlm_qa",
    "generic",
]


def system_prompt(*, dataset_name: RubricName) -> str:
    """Return the judge instructions matching the selected scoring contract."""
    return generic_rubric.GENERIC_RATER_SYSTEM_PROMPT if dataset_name == "generic" else rubrics.RATER_SYSTEM_PROMPT


def build_rater_user_content(
    *, source_image_path: Path, rendered_image_path: Path, dataset_name: RubricName
) -> list[dict[str, JsonValue]]:
    """Build the upstream reference/candidate comparison prompt."""
    builder = (
        generic_rubric.build_generic_rater_user_content
        if dataset_name == "generic"
        else rubrics.build_rater_user_content
    )
    return builder(
        source_image_path=source_image_path, rendered_image_path=rendered_image_path, dataset_name=dataset_name
    )


def aggregate_rating(
    parsed: dict[str, JsonValue],
    *,
    dataset_name: RubricName,
    candidate_inspection: dict[str, JsonValue],
    reference_image_path: str = "",
    candidate_image_path: str = "",
) -> dict[str, JsonValue]:
    """Score a successful render using the selected upstream weighting and caps."""
    scorer = generic_rubric.aggregate_generic_rating if dataset_name == "generic" else rubrics.aggregate_rating_v2
    return scorer(
        parsed,
        dataset_name=dataset_name,
        candidate_inspection=candidate_inspection,
        reference_image_path=reference_image_path,
        candidate_image_path=candidate_image_path,
    )


def build_render_failure_rating(status: str, *, dataset_name: RubricName) -> dict[str, JsonValue]:
    """Return the selected rubric's zero-reward result without calling the judge."""
    builder = (
        generic_rubric.build_generic_render_failure_rating
        if dataset_name == "generic"
        else rubrics.build_render_failure_rating
    )
    return builder(status, dataset_name=dataset_name)
