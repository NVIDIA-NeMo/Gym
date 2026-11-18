# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from resources_servers.question_bank_judge.answer_extraction import extract_last_boxed_answer


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"Earlier \boxed{2}; final \boxed{3}", "3"),
        (r"Final \boxed {\frac{1}{2}}", r"\frac{1}{2}"),
        ("Final \\boxed\n{3}", "3"),
        (r"\boxed{\{1,2\}}", r"\{1,2\}"),
        (r"\boxed{\frac{a+{b}}{c}}", r"\frac{a+{b}}{c}"),
        (r"\boxed{2} then \boxed{3", None),
        (r"\boxed{2} then \boxed {\frac{3}{4}", None),
        (r"\boxed{2} then \boxed{}", None),
        (r"\boxed{2} then \boxed{ }", None),
        (r"\boxed{2} then \boxed", None),
        (r"\boxed{2} then \boxed no argument", None),
        (r"\boxed{2} then \\boxed{3}", "2"),
        (r"\\boxed{3}", None),
        (r"\boxedvalue{3}", None),
        ("Nothing boxed", None),
    ],
)
def test_last_box_selection(response: str, expected: str | None) -> None:
    assert extract_last_boxed_answer(response) == expected
