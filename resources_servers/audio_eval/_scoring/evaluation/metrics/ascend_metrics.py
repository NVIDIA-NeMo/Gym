# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corpus-level ASCEND mixed and character error rates under both scoring protocols."""

from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


class ASCENDMetrics(BaseMetrics):
    """Pool the per-sample error counts of ``evaluate_ascend_sample`` into corpus-level rates."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("ASCENDMetrics supports only max_k=1. Run ASCEND with one greedy generation.")
        self.max_k = 1

    def reset(self):
        super().reset()
        self.mer_errors = 0
        self.mer_ref_units = 0
        self.cer_errors = 0
        self.cer_ref_chars = 0
        self.upstream_mer_errors = 0
        self.upstream_cer_errors = 0

    def _get_score_dict(self, prediction):
        return {"correct": bool(prediction["is_correct"])}

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("ASCENDMetrics expects exactly one prediction per sample.")
        super().update(predictions)
        prediction = predictions[0]
        self.mer_errors += int(prediction["mer_errors"])
        self.mer_ref_units += int(prediction["mer_ref_units"])
        self.cer_errors += int(prediction["cer_errors"])
        self.cer_ref_chars += int(prediction["cer_ref_chars"])
        self.upstream_mer_errors += int(prediction["mer_upstream_literal_errors"])
        self.upstream_cer_errors += int(prediction["cer_upstream_literal_errors"])
        predicted_answers = [(prediction.get("generation") or "").strip() or None]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        metrics = super().get_metrics()
        for values in metrics.values():
            values["success_rate"] = values.pop("correct", 0.0)
            if "no_answer" in values:
                values["no_answer"] = values["no_answer"] / 2.0
            # Both protocols share the reference, so the denominators are the same.
            if self.mer_ref_units > 0:
                values["mer"] = round(100.0 * self.mer_errors / self.mer_ref_units, 2)
                values["mer_upstream_literal"] = round(100.0 * self.upstream_mer_errors / self.mer_ref_units, 2)
            if self.cer_ref_chars > 0:
                values["cer"] = round(100.0 * self.cer_errors / self.cer_ref_chars, 2)
                values["cer_upstream_literal"] = round(100.0 * self.upstream_cer_errors / self.cer_ref_chars, 2)
            values["mer_errors"] = self.mer_errors
            values["mer_ref_units"] = self.mer_ref_units
            values["cer_errors"] = self.cer_errors
            values["cer_ref_chars"] = self.cer_ref_chars
            values["mer_upstream_literal_errors"] = self.upstream_mer_errors
            values["mer_upstream_literal_ref_units"] = self.mer_ref_units
            values["cer_upstream_literal_errors"] = self.upstream_cer_errors
            values["cer_upstream_literal_ref_chars"] = self.cer_ref_chars
        return metrics
