# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics aggregation for HALAS."""

from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


class HALASMetrics(BaseMetrics):
    """Corpus-level WER plus insertion-based hallucination proxy metrics."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("HALASMetrics supports only max_k=1. Run HALAS with one greedy generation.")
        self.max_k = 1
        self.wer_total_errors = 0
        self.wer_total_ref_words = 0
        self.substitutions = 0
        self.insertions = 0
        self.deletions = 0
        self.insertion_samples = 0
        self.pred_words = 0
        self.audio_seconds = 0.0

    def reset(self):
        super().reset()
        self.wer_total_errors = 0
        self.wer_total_ref_words = 0
        self.substitutions = 0
        self.insertions = 0
        self.deletions = 0
        self.insertion_samples = 0
        self.pred_words = 0
        self.audio_seconds = 0.0

    def _get_score_dict(self, prediction):
        return {"correct": bool(prediction.get("is_correct"))}

    def update_common_metrics(self, agg_dict):
        agg_dict["num_entries"] = self.total
        agg_dict["avg_tokens"] = int(self.avg_tokens / self.total) if self.total > 0 else 0
        if self.max_end_time > float("-inf") and self.min_start_time < float("inf"):
            agg_dict["gen_seconds"] = int(self.max_end_time - self.min_start_time)

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("HALASMetrics expects exactly one prediction per sample.")
        super().update(predictions)
        pred = predictions[0]

        self.wer_total_errors += int(pred.get("wer_errors") or 0)
        self.wer_total_ref_words += int(pred.get("wer_ref_words") or 0)
        self.substitutions += int(pred.get("wer_substitutions") or 0)
        self.insertions += int(pred.get("wer_insertions") or 0)
        self.deletions += int(pred.get("wer_deletions") or 0)
        self.insertion_samples += int(bool(pred.get("has_insertion")))
        self.pred_words += int(pred.get("pred_words") or 0)
        self.audio_seconds += float(pred.get("audio_duration") or 0.0)

        predicted_answers = [(pred.get("generation") or "").strip() or None]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        for _agg_mode, agg_metrics in metrics_dict.items():
            if "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]
            if "no_answer" in agg_metrics:
                agg_metrics["no_answer"] = agg_metrics["no_answer"] / 2.0

            if self.wer_total_ref_words > 0:
                agg_metrics["wer"] = round(100.0 * self.wer_total_errors / self.wer_total_ref_words, 2)
                agg_metrics["insertion_rate"] = round(100.0 * self.insertions / self.wer_total_ref_words, 2)
            if self.total > 0:
                agg_metrics["insertion_sample_rate"] = round(100.0 * self.insertion_samples / self.total, 2)
                agg_metrics["hallucination_proxy_rate"] = agg_metrics["insertion_sample_rate"]
            if self.audio_seconds > 0:
                agg_metrics["pred_words_per_minute"] = round(self.pred_words / (self.audio_seconds / 60.0), 2)

            agg_metrics["substitutions"] = self.substitutions
            agg_metrics["insertions"] = self.insertions
            agg_metrics["deletions"] = self.deletions
            agg_metrics["ref_words"] = self.wer_total_ref_words

        return metrics_dict
