# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.indic.frontier_math.normalization import normalized_grade, recovery_candidates


def _failed_row(*, status: str, extracted: str | None, text: str = "") -> dict:
    return {
        "reward": 0.0,
        "grading_status": status,
        "expected_answer": "1234",
        "answer_type": "integer",
        "extracted_answer": extracted,
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }
            ]
        },
    }


@pytest.mark.parametrize(
    ("row", "method"),
    [
        (_failed_row(status="missing_answer", extracted=None, text=r"\boxed{1234} then \boxed{"), "last_complete_box"),
        (_failed_row(status="invalid_expression", extracted="n=1234"), "top_level_rhs"),
        (_failed_row(status="invalid_expression", extracted=r"1\,234"), "latex_format"),
    ],
)
def test_normalized_grade_recovers_formatting_only(row: dict, method: str) -> None:
    result = normalized_grade(row)
    assert result.reward == 1.0
    assert result.recovery_method == method


def test_normalized_grade_does_not_search_for_gold_or_override_wrong_answer() -> None:
    embedded_gold = _failed_row(status="invalid_expression", extracted="the answer is 1234")
    assert normalized_grade(embedded_gold).reward == 0.0
    wrong_final = _failed_row(status="incorrect", extracted="999", text=r"\boxed{1234}\boxed{999}")
    assert normalized_grade(wrong_final).reward == 0.0
    assert list(recovery_candidates(wrong_final)) == []
