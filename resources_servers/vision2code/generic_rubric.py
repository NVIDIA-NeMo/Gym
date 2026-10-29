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

# Portions Copyright 2026 Vision2Code Authors (Apache-2.0).
# Focused port; see NOTICE and tests/fixtures/provenance.json.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from resources_servers.vision2code.rubrics import (
    SCORE_MAX,
    SCORE_MIN,
    _comparison_content,
    _render_failure_result,
    _round_tenth,
    _score_categories,
)


GENERIC_RUBRIC_VERSION = "generic_recreation_rubric_v1_equal_weight"


def _category(category_id: str, label: str, description: str) -> Dict[str, Any]:
    return {
        "id": category_id,
        "label": label,
        "weight": 0.25,
        "description": description,
    }


GENERIC_RUBRIC = {
    "rubric_dataset": "generic",
    "categories": [
        _category(
            "core_information_fidelity",
            "Core Information Fidelity",
            "Preserves important visible information, values, objects, relations, or semantics.",
        ),
        _category(
            "structure_layout_fidelity",
            "Structure and Layout Fidelity",
            "Preserves overall layout, geometry, chart or diagram structure, document structure, and spatial arrangement.",
        ),
        _category(
            "text_annotation_accuracy",
            "Text and Annotation Accuracy",
            "Preserves readable labels, numbers, symbols, legends, annotations, and text.",
        ),
        _category(
            "visual_completeness_cleanliness",
            "Visual Completeness and Cleanliness",
            "The output is complete, unclipped, readable, nonblank, and visually clean.",
        ),
    ],
}


GENERIC_RATER_SYSTEM_PROMPT = """\
You are a rigorous but calibrated evaluator for image recreation quality.
Compare the reference image against the candidate rendered image.
Judge only what is visibly present in the images.

Use the rubric provided by the user message. It applies to every dataset.
Be strict about semantically wrong recreations, but do not over-focus on small cosmetic differences.

Scoring anchors on a 0.0 to 5.0 scale:
- 5.0 = near-exact or clearly excellent
- 4.0 = strong recreation with minor issues
- 3.0 = good but with noticeable structural, text, or semantic problems
- 2.0 = partial recreation with important errors
- 1.0 = major mismatch
- 0.0 = missing, unreadable, blank, or fundamentally broken

Return JSON only. Do not include markdown fences.
"""


def generic_rubric_markdown() -> str:
    lines = ["Rubric: generic image recreation baseline"]
    for category in GENERIC_RUBRIC["categories"]:
        lines.append(f"- {category['id']} (weight {category['weight']:.2f}): {category['description']}")
    return "\n".join(lines)


def build_generic_rater_user_content(
    *,
    source_image_path: Path,
    rendered_image_path: Path,
    metadata: Optional[Mapping[str, Any]] = None,
    dataset_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build the generic judge prompt with reference and candidate images."""
    metadata_payload = dict(metadata or {})
    prompt_text = (
        "Evaluate the candidate recreation against the reference image using this rubric.\n\n"
        f"Rubric version: {GENERIC_RUBRIC_VERSION}\n"
        "Rubric dataset: generic\n"
        f"Source dataset name, if known: {dataset_name or ''}\n\n"
        "Categories:\n"
        f"{generic_rubric_markdown()}\n\n"
        "Reference metadata:\n"
        f"{json.dumps(metadata_payload, indent=2, ensure_ascii=False)}\n\n"
        "Return JSON with exactly these top-level keys:\n"
        '1) "category_scores"\n'
        '2) "rationales"\n'
        '3) "strengths"\n'
        '4) "issues"\n'
        '5) "overall_summary"\n\n'
        "Rules:\n"
        "- 'category_scores' must contain every category id with a numeric score from 0.0 to 5.0.\n"
        "- Use decimals in increments of 0.1 when needed.\n"
        "- 'rationales' must contain the same keys, with one short rationale per category.\n"
        "- 'strengths' must be a short list of 1 to 3 bullets.\n"
        "- 'issues' must be a short list of 1 to 5 bullets.\n"
        "- 'overall_summary' must be 1 to 3 concise sentences.\n"
        "- Use the same four generic categories for every dataset.\n"
        "- Do not compute the final 0.0 to 5.0 rating yourself; provide category scores only.\n"
    )

    return _comparison_content(
        source_image_path=source_image_path, rendered_image_path=rendered_image_path, prompt_text=prompt_text
    )


def aggregate_generic_rating(
    parsed: Dict[str, Any],
    *,
    dataset_name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    execution_status: str = "ok",
    execution_error_present: bool = False,
    candidate_inspection: Optional[Mapping[str, Any]] = None,
    reference_image_path: str = "",
    candidate_image_path: str = "",
) -> Dict[str, Any]:
    """Average the four generic category scores without dataset caps."""
    category_results, raw_score = _score_categories(parsed, categories=GENERIC_RUBRIC["categories"], critical_count=0)

    strengths = [str(item).strip() for item in parsed.get("strengths", []) if str(item).strip()]
    issues = [str(item).strip() for item in parsed.get("issues", []) if str(item).strip()]
    overall_summary = str(parsed.get("overall_summary", "")).strip()
    final_rating = _round_tenth(max(SCORE_MIN, min(SCORE_MAX, raw_score)))

    if execution_status != "ok":
        issues = [f"Render failed before visual evaluation: {execution_status}"]
        overall_summary = (
            "The generated code did not produce a valid rendered image, so the sample receives the minimum rating."
        )
        raw_score = 0.0
        final_rating = 0.0

    return {
        "rubric_version": GENERIC_RUBRIC_VERSION,
        "rubric_dataset": "generic",
        "source_dataset": dataset_name or dict(metadata or {}).get("dataset", ""),
        "status": "ok" if execution_status == "ok" else "render_failed",
        "execution_status": execution_status,
        "execution_detail": "" if execution_status == "ok" else execution_status,
        "execution_error_present": execution_error_present,
        "reference_image_path": reference_image_path,
        "candidate_image_path": candidate_image_path,
        "candidate_inspection": dict(candidate_inspection or {}),
        "category_scores": category_results,
        "raw_score_0_to_5": round(raw_score, 4),
        "raw_visual_score_0_to_5": round(raw_score, 4),
        "provisional_rating_0_to_5": final_rating,
        "applied_caps": [],
        "final_rating_0_to_5": final_rating,
        "strengths": strengths[:3],
        "issues": issues[:5],
        "overall_summary": overall_summary,
    }


def build_generic_render_failure_rating(
    status: str,
    *,
    dataset_name: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    execution_error_present: bool = True,
    reference_image_path: str = "",
    candidate_image_path: str = "",
    candidate_inspection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a zero-score generic result for a failed render."""
    return {
        **_render_failure_result(
            status,
            categories=GENERIC_RUBRIC["categories"],
            critical_count=0,
            execution_error_present=execution_error_present,
            reference_image_path=reference_image_path,
            candidate_image_path=candidate_image_path,
            candidate_inspection=candidate_inspection,
        ),
        "rubric_version": GENERIC_RUBRIC_VERSION,
        "rubric_dataset": "generic",
        "source_dataset": dataset_name or dict(metadata or {}).get("dataset", ""),
        "applied_caps": [],
    }
