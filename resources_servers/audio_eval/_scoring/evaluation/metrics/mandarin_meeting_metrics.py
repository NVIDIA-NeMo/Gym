# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corpus cpCER aggregation for speaker-attributed Mandarin meetings."""

from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


class MandarinMeetingMetrics(BaseMetrics):
    """Session-level cpCER pooled over the corpus (``cpcer``) and averaged over sessions (``cpcer_macro``), in percent."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("MandarinMeetingMetrics supports one greedy generation only")
        self.max_k = 1

    def reset(self):
        super().reset()
        self.errors = 0
        self.substitutions = 0
        self.insertions = 0
        self.deletions = 0
        self.ref_chars = 0
        self.session_scores: list[float] = []

    def _get_score_dict(self, prediction):
        return {"correct": bool(prediction["is_correct"])}

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("MandarinMeetingMetrics expects exactly one prediction per sample")
        super().update(predictions)
        prediction = predictions[0]
        self.errors += int(prediction["cpcer_errors"])
        self.substitutions += int(prediction["cpcer_substitutions"])
        self.insertions += int(prediction["cpcer_insertions"])
        self.deletions += int(prediction["cpcer_deletions"])
        self.ref_chars += int(prediction["cpcer_ref_chars"])
        self.session_scores.append(float(prediction["cpcer"]))
        answer = (prediction.get("generation") or "").strip() or None
        self._compute_pass_at_k(predictions, [answer])
        self._compute_majority_at_k(predictions, [answer])

    def get_metrics(self):
        metrics = super().get_metrics()
        for values in metrics.values():
            values["success_rate"] = values.pop("correct", 0.0)
            if "no_answer" in values:
                # The base class counts an empty answer twice (pass@1 and majority@1); halve it as the other
                # audio benchmarks do.
                values["no_answer"] /= 2.0
            if self.ref_chars:
                values["cpcer"] = round(100.0 * self.errors / self.ref_chars, 2)
                values["cpcer_macro"] = round(100.0 * sum(self.session_scores) / len(self.session_scores), 2)
            values["cpcer_errors"] = self.errors
            values["cpcer_substitutions"] = self.substitutions
            values["cpcer_insertions"] = self.insertions
            values["cpcer_deletions"] = self.deletions
            values["cpcer_ref_chars"] = self.ref_chars
        return metrics
