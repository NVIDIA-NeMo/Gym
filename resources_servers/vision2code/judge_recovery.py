# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2026 Vision2Code Authors
# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/image2code/vision2code (Apache-2.0).
# Revision: 000f2e162dd62fac5637c718e018d7123b3aff1e.
# Sources: vision2code/evaluation/dataset_rubrics.py, vision2code/evaluation/generic_rubric.py,
# and vision2code/benchmark/run_benchmark.py. See NOTICE.
"""Bounded judge recovery using the original images and selected rubric."""

from pydantic import JsonValue

from resources_servers.vision2code import generic_rubric, rubrics
from resources_servers.vision2code.evaluation import RubricName


def category_ids(dataset_name: RubricName) -> list[str]:
    """Return the required score keys for the selected rubric, including variants."""
    definition = generic_rubric.GENERIC_RUBRIC if dataset_name == "generic" else rubrics.DATASET_RUBRICS[dataset_name]
    return [category["id"] for category in definition["categories"]]


def repair_prompt(raw_response: str, error_message: str, *, dataset_name: RubricName) -> str:
    """Ask the judge to correct malformed JSON without changing the rubric."""
    expected_keys = ", ".join(category_ids(dataset_name))
    return (
        "The previous evaluation response was invalid.\n"
        f"Validation error: {error_message}\n\n"
        "Return corrected JSON only with these top-level keys:\n"
        '  "category_scores", "rationales", "strengths", "issues", "overall_summary"\n'
        "The 'category_scores' object must include every category id exactly once.\n"
        "Each category score must be numeric in the range 0.0 to 5.0, preferably in 0.1 increments.\n"
        f"Required category ids: {expected_keys}\n\n"
        "Previous response:\n"
        f"{raw_response}"
    )


REJUDGE_PROMPT = (
    "Produce a fresh complete evaluation of the two images using the same rubric. "
    "Return only the required JSON. Each rationale and summary must be one short "
    "sentence, at most 240 characters. Do not repeat comparisons or enumerate "
    "all chart values. Recheck that each score agrees with its visual evidence."
)


def recovery_schema(dataset_name: RubricName) -> dict[str, JsonValue]:
    """Constrain recovery text length and score keys; never salvage partial scores."""
    keys = category_ids(dataset_name)
    short_text = {"type": "string", "pattern": r"^[^\n\r]{1,240}$"}

    def obj(properties: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    return obj(
        {
            "category_scores": obj({k: {"type": "number", "minimum": 0, "maximum": 5} for k in keys}),
            "rationales": obj({k: dict(short_text) for k in keys}),
            "strengths": {"type": "array", "items": short_text, "maxItems": 3},
            "issues": {"type": "array", "items": short_text, "maxItems": 5},
            "overall_summary": dict(short_text),
        }
    )
