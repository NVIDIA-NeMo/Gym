# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned VoiceBench A–D parsing and legacy random-guess diagnostics."""

import random

from pydantic import JsonValue

from resources_servers.voicebench.scorers.base import ScoringResult
from resources_servers.voicebench.vendor.mcq import MCQEvaluator


def score(answer: str, metadata: dict[str, JsonValue], *, rng: random.Random) -> ScoringResult:
    """Score a final answer while retaining legacy guesses in the accuracy count."""
    reference = metadata.get("reference")
    if not isinstance(reference, str):
        raise ValueError("VoiceBench MCQ requires a string reference letter")
    prediction = MCQEvaluator().extract_answer(answer)
    failed = prediction is None
    if failed:
        prediction = rng.choice(["A", "B", "C", "D"])
    correct = prediction == reference
    return ScoringResult(
        reward=float(correct),
        metrics={"correct": float(correct), "total": 1.0, "parse_failure": float(failed)},
        details={
            "prediction": prediction,
            "reference": reference,
            "parse_failure": failed,
            "legacy_random_fallback": failed,
        },
    )
