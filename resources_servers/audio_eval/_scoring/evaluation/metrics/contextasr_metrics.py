# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA-NeMo/Skills (Apache-2.0), nemo_skills/evaluation/metrics/contextasr_metrics.py
# at commit da85a881d972e6fec847b90cf553a0bf9bf10638: https://github.com/NVIDIA-NeMo/Skills
# Modifications: the Skills base-class import is replaced by this package's `BaseMetrics`; the terminal-display hooks
# (`evaluations_to_print`, `metrics_to_print`) and `get_incorrect_sample` are removed; `max_k` other than 1 is
# rejected when the object is built, `reset` clears the pooled counts, and `no_answer` is halved (see `get_metrics`).

"""Metrics aggregation for ContextASR-Bench evaluation.

Computes corpus-level:
- WER: total word errors / total reference words
- NE-WER: total entity word errors / total entity reference words
- NE-FNR: 1 - (total entity hits / total entities)
"""

from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


class ContextASRMetrics(BaseMetrics):
    """Metrics class for ContextASR-Bench with corpus-level WER, NE-WER, NE-FNR."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        """Initialize accumulators for corpus-level WER, NE-WER, and NE-FNR."""
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError(
                "ContextASRMetrics supports only max_k=1. Run ContextASR-Bench with one greedy generation."
            )
        self.max_k = 1
        self._reset_counts()

    def _reset_counts(self):
        self.wer_total_errors = 0
        self.wer_total_ref_words = 0

        self.ne_wer_total_errors = 0
        self.ne_wer_total_ref_words = 0

        self.ne_fnr_total_hits = 0
        self.ne_fnr_total_entities = 0

    def reset(self):
        """Clear the pooled counts together with the base accumulators."""
        super().reset()
        self._reset_counts()

    def _get_score_dict(self, prediction):
        """Extract the binary correctness score from a prediction (WER < 0.5).

        Uses the standard ``correct`` key so the base class populates
        ``pass@k["correct"]``, which ``get_metrics`` then surfaces as
        ``success_rate``.
        """
        return {"correct": prediction["is_correct"]}

    def update_common_metrics(self, agg_dict):
        """Populate num_entries, avg_tokens, and gen_seconds into the aggregation dict."""
        agg_dict["num_entries"] = self.total
        agg_dict["avg_tokens"] = int(self.avg_tokens / self.total) if self.total > 0 else 0
        if self.max_end_time > float("-inf") and self.min_start_time < float("inf"):
            agg_dict["gen_seconds"] = int(self.max_end_time - self.min_start_time)

    def update(self, predictions):
        """Accumulate per-sample error counts for corpus-level metric computation.

        ContextASR-Bench corpus-level WER/NE-WER/NE-FNR are defined per single
        hypothesis: there is no canonical way to combine WER across k
        hypotheses without reference comparison (which would defeat pass@k).
        Multi-generation aggregation is therefore not supported here -- run
        with a single greedy generation.
        """
        if len(predictions) != 1:
            raise ValueError(
                f"ContextASRMetrics expects exactly 1 generation per sample, "
                f"got {len(predictions)}. Run with a single greedy generation "
                f"(num_random_seeds=1) for ContextASR-Bench."
            )
        super().update(predictions)

        pred = predictions[0]
        predicted_answers = [(pred.get("generation") or "").strip() or None]

        self.wer_total_errors += pred["wer_errors"]
        self.wer_total_ref_words += pred["wer_ref_words"]
        self.ne_wer_total_errors += pred["ne_wer_errors"]
        self.ne_wer_total_ref_words += pred["ne_wer_ref_words"]
        self.ne_fnr_total_hits += pred["ne_fnr_hits"]
        self.ne_fnr_total_entities += pred["ne_fnr_total"]

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        """Compute corpus-level WER, NE-WER, NE-FNR percentages from accumulated counts."""
        metrics_dict = super().get_metrics()

        for _agg_mode, agg_metrics in metrics_dict.items():
            if "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]
            if "no_answer" in agg_metrics:
                # The base class counts an empty answer twice (as in the other audio metrics), so halve it.
                agg_metrics["no_answer"] = agg_metrics["no_answer"] / 2.0

            if self.wer_total_ref_words > 0:
                agg_metrics["wer"] = round(100.0 * self.wer_total_errors / self.wer_total_ref_words, 2)

            if self.ne_wer_total_ref_words > 0:
                agg_metrics["ne_wer"] = round(100.0 * self.ne_wer_total_errors / self.ne_wer_total_ref_words, 2)

            if self.ne_fnr_total_entities > 0:
                agg_metrics["ne_fnr"] = round(100.0 * (1.0 - self.ne_fnr_total_hits / self.ne_fnr_total_entities), 2)

        return metrics_dict
