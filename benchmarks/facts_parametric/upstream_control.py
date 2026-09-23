# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upstream control: the label-extraction and scoring helpers of the official FACTS Parametric starter notebook.

Copied verbatim (Apache 2.0) from Kaggle notebook ``yulongt/facts-parametric-benchmark-starter-code`` version 10
(cell "Helper Functions"); ``STARTER_CELL_SHA256`` pins the cell text. ``calibrate.py`` replays every grader receipt
through these functions and compares the result with the Gym verifier's labels and rewards.
"""

STARTER_NOTEBOOK = "yulongt/facts-parametric-benchmark-starter-code"
STARTER_VERSION = 10
STARTER_CELL_SHA256 = "ddb61f29a532e57d0dbf7701471b5aecbe3446fb7bfde5d33ccc6e44d138e033"  # pragma: allowlist secret


def extract_classification(judgment: str) -> str:
    """Extract the classification from the judgment text."""
    judgment = judgment.strip()

    if "INCORRECT" in judgment:
        return "INCORRECT"
    elif "MISTAKE" in judgment:
        return "MISTAKE"
    elif "CORRECT" in judgment:
        return "CORRECT"
    elif "NOT_ATTEMPTED" in judgment:
        return "NOT_ATTEMPTED"
    else:
        return "UNKNOWN"


def calculate_score(judgments: list) -> float:
    """Calculate a final score based on judgments.

    All judgments must be 'CORRECT' to get a score of 1.0, otherwise 0.0.
    """
    # All judgments must be CORRECT to get a score of 1.0
    if all(j == "CORRECT" for j in judgments):
        return 1.0
    else:
        return 0.0
