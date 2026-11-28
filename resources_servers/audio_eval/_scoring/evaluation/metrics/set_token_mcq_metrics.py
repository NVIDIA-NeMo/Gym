# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy aggregation for deterministic MMAR set-token scoring."""

from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


class SetTokenMCQMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool]:
        return {"accuracy": bool(prediction.get("symbolic_correct", False))}

    def update(self, predictions):
        super().update(predictions)
        predicted_answers = [pred["predicted_answer"] for pred in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)
