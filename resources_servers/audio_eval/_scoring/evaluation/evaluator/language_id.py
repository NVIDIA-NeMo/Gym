# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable exact-label language-identification evaluator."""

from collections.abc import Collection
from typing import Any

from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator


def evaluate_language_id_sample(
    data_point: dict[str, Any], valid_labels: Collection[str] | None = None
) -> dict[str, Any]:
    """Grade one top-1 language label after outer-whitespace trimming only."""
    expected = str(data_point["expected_answer"])
    if valid_labels is not None and expected not in valid_labels:
        raise ValueError(f"Non-canonical language-ID reference label: {expected!r}")
    predicted = (data_point.get("generation") or "").strip()
    extra_fields = data_point.get("extra_fields") or {}
    label_group = (
        data_point.get("label_group")
        or data_point.get("lang_group")
        or extra_fields.get("label_group")
        or extra_fields.get("lang_group")
    )
    result = {
        "generation": predicted,
        "predicted_label": predicted,
        "lid_correct": predicted == expected,
        "is_correct": predicted == expected,
        "label": expected,
        "label_group": str(label_group) if label_group is not None else None,
    }
    if valid_labels is not None:
        result["valid_label"] = predicted in valid_labels
    return result


class LanguageIDEvaluator(BaseEvaluator):
    def __init__(self, config: dict[str, Any], num_parallel_requests: int = 10):
        super().__init__(config, num_parallel_requests)
        valid_labels = config.get("valid_labels") if hasattr(config, "get") else None
        self.valid_labels = frozenset(valid_labels) if valid_labels else None

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_language_id_sample(data_point, valid_labels=self.valid_labels)
