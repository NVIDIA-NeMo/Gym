# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from dataclasses import asdict
from unittest.mock import patch

import pytest

from benchmarks.indic.frontier_math.normalization import normalized_grade, recovery_candidates
from resources_servers.frontiermath.grading import grade_answer


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


@pytest.mark.parametrize(
    ("extracted", "reward"),
    [
        (r"x=\frac{1}{2}", 1.0),
        (r"\dfrac{1}{2}", 1.0),
        (r"x=\frac{2}{3}", 0.0),
    ],
)
def test_symbolic_recovery_uses_exact_verifier(extracted: str, reward: float) -> None:
    row = _failed_row(status="invalid_expression", extracted=extracted)
    row.update(answer_type="expression", expected_answer=r"\frac{1}{2}")
    assert normalized_grade(row).reward == reward


@pytest.mark.parametrize(
    ("answer", "reward"),
    [
        ("n=214", 1.0),
        ("x = 214", 1.0),
        ("n=215", 0.0),
        ("2n=214", 0.0),
        ("n!=214", 0.0),
        ("n<=214", 0.0),
        ("n>=214", 0.0),
        ("n+1=214", 0.0),
        ("n^2=214", 0.0),
        ("n=107=214", 0.0),
        (r"n\ne=214", 0.0),
        (r"\frac{n}{2}=214", 0.0),
    ],
)
def test_assignment_recovery_from_real_strict_grades(answer: str, reward: float) -> None:
    text = rf"\boxed{{{answer}}}"
    strict = grade_answer(expected_answer="214", answer_type="integer", generated_answer=text)
    assert strict.reward == 0.0
    row = _failed_row(status=strict.grading_status, extracted=strict.extracted_answer, text=text)
    row.update(asdict(strict), expected_answer="214")
    assert normalized_grade(row).reward == reward


def test_assignment_recovery_preserves_exact_factorials() -> None:
    text = r"\boxed{n=5!}"
    strict = grade_answer(expected_answer="120", answer_type="integer", generated_answer=text)
    assert strict.reward == 0.0
    row = _failed_row(status=strict.grading_status, extracted=strict.extracted_answer, text=text)
    row.update(asdict(strict), expected_answer="120")
    assert normalized_grade(row).reward == 1.0


@pytest.mark.parametrize("failure", ["spawn", "timeout", "crash", "invalid_json", "invalid_grade"])
def test_normalization_preserves_verifier_failures(failure: str) -> None:
    row = _failed_row(status="invalid_expression", extracted=r"x=\frac{1}{2}")
    row.update(answer_type="expression", expected_answer=r"\frac{1}{2}")
    stdout = "not json" if failure == "invalid_json" else "{}"
    process = subprocess.CompletedProcess([], 1 if failure == "crash" else 0, stdout=stdout)
    error = OSError("Unavailable") if failure == "spawn" else subprocess.TimeoutExpired([], 5)
    with patch(
        "benchmarks.indic.frontier_math.normalization.subprocess.run",
        side_effect=error if failure in {"spawn", "timeout"} else None,
        return_value=process,
    ):
        result = normalized_grade(row)
    assert result.failure_reason
    assert result.recovered_answer is None
