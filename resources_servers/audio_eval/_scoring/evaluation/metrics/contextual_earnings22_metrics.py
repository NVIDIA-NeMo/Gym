# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA-NeMo/Skills (Apache-2.0), nemo_skills/evaluation/metrics/contextual_earnings22_metrics.py
# at commit da85a881d972e6fec847b90cf553a0bf9bf10638: https://github.com/NVIDIA-NeMo/Skills
# Modifications: imports point at this repository's `audio_eval` scorer package; the console-display methods
# (`evaluations_to_print`, `metrics_to_print`) and `get_incorrect_sample` are removed; `no_answer` is halved, as in
# the other audio_eval metrics; the rest is unchanged.

"""Metrics aggregation for Contextual Earnings-22 evaluation.

Computes corpus-level:

- WER:               sum(wer_errors) / sum(wer_ref_words)
- keyword_precision: sum(tp) / (sum(tp) + sum(fp))
- keyword_recall:    sum(tp) / (sum(tp) + sum(fn))
- keyword_f1:        harmonic mean of corpus precision and recall
"""

from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


class ContextualEarnings22Metrics(BaseMetrics):
    """Corpus-level WER + keyword Precision / Recall / F1 for Contextual Earnings-22."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        """Initialize accumulators for corpus-level WER and keyword P/R/F1.

        Only ``max_k == 1`` is supported: corpus-level WER and keyword
        precision/recall/F1 are not well-defined across multiple hypotheses
        per sample, so ``ContextualEarnings22Metrics`` rejects multi-generation
        configurations up front rather than partially mutating state on the
        first ``update()`` call.
        """
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError(
                f"ContextualEarnings22Metrics supports only max_k=1, got {max_k}. "
                f"Run with a single greedy generation (num_random_seeds=1) for Contextual Earnings-22."
            )
        self.max_k = 1

        self.wer_total_errors = 0
        self.wer_total_ref_words = 0

        self.kw_total_tp = 0
        self.kw_total_fp = 0
        self.kw_total_fn = 0

    def reset(self):
        """Reset all accumulators (including custom ones) between runs."""
        super().reset()
        self.wer_total_errors = 0
        self.wer_total_ref_words = 0
        self.kw_total_tp = 0
        self.kw_total_fp = 0
        self.kw_total_fn = 0

    def _get_score_dict(self, prediction):
        """Extract the binary correctness score (WER < 0.5) for the base class."""
        return {"correct": prediction["is_correct"]}

    def update_common_metrics(self, agg_dict):
        """Populate num_entries, avg_tokens, and gen_seconds into the aggregation dict."""
        agg_dict["num_entries"] = self.total
        agg_dict["avg_tokens"] = int(self.avg_tokens / self.total) if self.total > 0 else 0
        if self.max_end_time > float("-inf") and self.min_start_time < float("inf"):
            agg_dict["gen_seconds"] = int(self.max_end_time - self.min_start_time)

    def update(self, predictions):
        """Accumulate per-sample raw counts for corpus-level metric computation.

        Contextual Earnings-22 corpus-level WER and keyword P/R/F1 are defined
        per single hypothesis: there is no canonical way to combine these
        metrics across k hypotheses without reference comparison (which would
        defeat pass@k). Multi-generation aggregation is therefore not supported
        here -- run with a single greedy generation. Validation happens before
        ``super().update()`` so an invalid call cannot partially mutate state.
        """
        if len(predictions) != 1:
            raise ValueError(
                f"ContextualEarnings22Metrics expects exactly 1 generation per sample, "
                f"got {len(predictions)}. Run with a single greedy generation "
                f"(num_random_seeds=1) for Contextual Earnings-22."
            )

        super().update(predictions)

        pred = predictions[0]
        predicted_answers = [pred["generation"].strip() or None]

        self.wer_total_errors += pred["wer_errors"]
        self.wer_total_ref_words += pred["wer_ref_words"]
        self.kw_total_tp += pred["keyword_tp"]
        self.kw_total_fp += pred["keyword_fp"]
        self.kw_total_fn += pred["keyword_fn"]

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        """Compute corpus-level WER and keyword micro P/R/F1 from accumulated counts."""
        metrics_dict = super().get_metrics()

        for _agg_mode, agg_metrics in metrics_dict.items():
            if "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]
            if "no_answer" in agg_metrics:
                # The base metrics double-count empty answers; halve as the other audio_eval metrics do.
                agg_metrics["no_answer"] = agg_metrics["no_answer"] / 2.0

            if self.wer_total_ref_words > 0:
                agg_metrics["wer"] = round(100.0 * self.wer_total_errors / self.wer_total_ref_words, 2)

            tp_plus_fp = self.kw_total_tp + self.kw_total_fp
            tp_plus_fn = self.kw_total_tp + self.kw_total_fn

            if tp_plus_fp > 0:
                precision = self.kw_total_tp / tp_plus_fp
                agg_metrics["keyword_precision"] = round(100.0 * precision, 2)
            else:
                precision = None

            if tp_plus_fn > 0:
                recall = self.kw_total_tp / tp_plus_fn
                agg_metrics["keyword_recall"] = round(100.0 * recall, 2)
            else:
                recall = None

            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2.0 * precision * recall / (precision + recall)
                agg_metrics["keyword_f1"] = round(100.0 * f1, 2)
            elif tp_plus_fp > 0 or tp_plus_fn > 0:
                agg_metrics["keyword_f1"] = 0.0

        return metrics_dict
