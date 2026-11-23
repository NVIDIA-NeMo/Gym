# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The verifier semantics against goldens produced by upstream EnterpriseOps-Gym's own code.

``verifier_golden.json`` holds what upstream ``benchmark/verifier.py`` returns for each case; regenerate it with
``generate_verifier_golden.py`` and an upstream checkout.
"""

import asyncio
import json
from pathlib import Path

import pytest

from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.enterpriseops_gym.app import VERIFIER_FIXTURE
from resources_servers.enterpriseops_gym.verifier import compare, extract_value


GOLDEN = json.loads((Path(__file__).parent / "verifier_golden.json").read_text())


def _number_and_numeric_string(case: dict) -> bool:
    values = (case["actual"], case["expected"])
    has_number = any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    has_numeric_string = any(
        isinstance(v, str) and v.strip().lstrip("-").replace(".", "", 1).isdigit() for v in values
    )
    return has_number and has_numeric_string


@pytest.mark.parametrize("case", GOLDEN["extraction"], ids=lambda case: json.dumps(case["answer"])[:60])
def test_extraction_matches_upstream(case):
    assert extract_value(case["answer"]) == case["value"]


def test_comparison_matches_upstream_except_numbers_against_numeric_strings():
    deviations = []
    for case in GOLDEN["comparison"]:
        ours = compare(case["actual"], case["expected"], case["comparison_type"])
        if ours == case["passed"]:
            continue
        deviations.append(case)
        # The one deviation: equality and ordering read a numeric string as the number it spells.
        assert case["comparison_type"] != "contains", case
        assert _number_and_numeric_string(case), case
        assert ours is True and case["passed"] is False, case
    assert len(deviations) == 21
    assert {"actual": 1, "expected": "1", "comparison_type": "equals", "passed": False} in deviations


@pytest.mark.parametrize(
    ("actual", "expected", "comparison_type", "passed"),
    [
        (1, "1", "equals", True),
        ("2", 2, "equals", True),
        (1, "1.0", "equals", True),
        (1, "x", "equals", False),
        (True, 1, "equals", True),
        (True, "1", "equals", False),
        (3, "2", "greater_than", True),
        ("a", 1, "greater_than", False),
        (5, 5, "contains", False),
        ("in_progress status", "in_progress", "contains", True),
    ],
)
def test_comparison_semantics(actual, expected, comparison_type, passed):
    assert compare(actual, expected, comparison_type) is passed


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
        )
    )
