# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned four-category VoiceBench BBH parsing with the supplied legacy random stream."""

import random

from pydantic import JsonValue

from resources_servers.voicebench.scorers.base import ScoringResult
from resources_servers.voicebench.vendor.bbh import BBHEvaluator
from resources_servers.voicebench.vendor.randomness import using_rng


def score(answer: str, metadata: dict[str, JsonValue], *, rng: random.Random) -> ScoringResult:
    """Dispatch by the original source task ID and preserve legacy parser quirks."""
    source_id = metadata.get("id") or metadata.get("sample_id")
    if not isinstance(source_id, str):
        raise ValueError("VoiceBench BBH requires its original task ID or sample_id")
    categories = ("sports_understanding", "hyperbaton", "navigate", "web_of_lies")
    category = next((name for name in categories if name in source_id), None)
    if category is None:
        raise ValueError(f"Unsupported VoiceBench BBH task ID: {source_id!r}")
    reference = metadata.get("reference")
    labels = {"yes": 1, "no": 0, "(a)": 0, "(b)": 1}
    if not isinstance(reference, str) or reference.lower() not in labels:
        raise ValueError("VoiceBench BBH requires reference yes/no/(a)/(b)")
    with using_rng(rng) as context:
        prediction = BBHEvaluator().extract_answer(answer, source_id)
    failed = context.draws > 0
    expected = labels[reference.lower()]
    correct = prediction == expected
    return ScoringResult(
        reward=float(correct),
        metrics={"correct": float(correct), "total": 1.0, "parse_failure": float(failed)},
        details={
            "category": category,
            "source_id": source_id,
            "prediction": prediction,
            "reference": reference,
            "parse_failure": failed,
            "legacy_random_fallback": failed,
        },
    )
