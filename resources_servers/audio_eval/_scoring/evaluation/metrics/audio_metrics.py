# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA-NeMo/Skills (Apache-2.0), nemo_skills/evaluation/metrics/audio_metrics.py
# at commit da85a881d972e6fec847b90cf553a0bf9bf10638: https://github.com/NVIDIA-NeMo/Skills

"""Audio metrics: Metrics aggregation for audio evaluation tasks.

This module provides comprehensive metrics tracking and aggregation for various
audio-related evaluation tasks. It supports automatic metrics (WER, CER, BLEU, etc.)
as well as judge-based evaluation for open-ended audio tasks.

Supported Metrics:
- WER: Word Error Rate (standard ASR metric)
- WER_C: WER with capitalization
- WER_PC: WER with punctuation and capitalization
- PER: Punctuation Error Rate
- BLEU: Translation quality metric
- CER: Character Error Rate (for character-level evaluation)
- Hallucination Rate: Detection of hallucinated content
- PC Rate: Punctuation/Capitalization recovery rate

The metrics class is designed to be extensible, allowing easy addition of
audio-specific metrics as the field evolves.
"""

import logging
from typing import Any

from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


LOG = logging.getLogger(__name__)


def compute_corpus_bleu(
    hyps: list[str],
    refs: list[str],
    tokenizes: list[str],
) -> float:
    """Compute corpus BLEU, bucketing by sacrebleu tokenizer.

    Each (hyp, ref) is grouped by the tokenizer resolved at sentence-BLEU
    time, so ja/zh/ko aren't silently scored under the default 13a tokenizer.
    Single-tokenizer runs are one corpus_bleu call; mixed runs weighted-average
    per-tokenizer corpus_bleu by sample count.
    """
    from sacrebleu import corpus_bleu

    groups: dict[str, tuple[list[str], list[str]]] = {}
    for hyp, ref, tokenize in zip(hyps, refs, tokenizes, strict=True):
        bucket = groups.setdefault(tokenize, ([], []))
        bucket[0].append(hyp)
        bucket[1].append(ref)

    if len(groups) == 1:
        tokenize, (group_hyps, group_refs) = next(iter(groups.items()))
        return corpus_bleu(hypotheses=group_hyps, references=[group_refs], tokenize=tokenize).score

    weighted_sum = 0.0
    total = 0
    for tokenize, (group_hyps, group_refs) in groups.items():
        score = corpus_bleu(hypotheses=group_hyps, references=[group_refs], tokenize=tokenize).score
        weighted_sum += score * len(group_hyps)
        total += len(group_hyps)
    return weighted_sum / total


class AudioMetrics(BaseMetrics):
    """Metrics class for audio evaluation tasks.

    This class tracks and aggregates various audio-specific metrics including
    error rates (WER, CER, PER), quality scores (BLEU), and advanced metrics
    like hallucination detection. It extends BaseMetrics to provide consistent
    metric computation and reporting across different audio tasks.
    """

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        """Initialize audio metrics with tracking lists for all supported metrics.

        Args:
            compute_no_answer: Whether to compute no_answer statistics
            max_k: Maximum k for pass@k and majority@k evaluation
        """
        super().__init__(compute_no_answer=compute_no_answer)
        self.max_k = max_k

        self.wer_scores = []

        # Corpus-level WER accumulators (total errors / total ref words)
        self.wer_total_errors = 0
        self.wer_total_ref_words = 0
        self.wer_total_substitutions = 0
        self.wer_total_insertions = 0
        self.wer_total_deletions = 0

        # Corpus-level error rates per explicit ``subset_for_metrics`` tag.
        # Multilingual benchmarks use locales/language pairs as subsets.
        self.wer_subset_totals: dict[str, dict[str, int]] = {}

        # Corpus-level cpWER accumulators (multispeaker ASR)
        self.cpwer_total_errors = 0
        self.cpwer_total_ref_words = 0
        self.cpwer_total_substitutions = 0
        self.cpwer_total_insertions = 0
        self.cpwer_total_deletions = 0
        self.cpwer_scores = []

        # Per-subset cpWER accumulators, keyed by each record's
        # ``subset_for_metrics`` tag. This lets a *merged* benchmark (e.g.
        # ``multispeaker-sot.under20s``, whose test.jsonl concatenates several
        # eval sets) still report a cpWER breakdown per member set in the single
        # metrics.json, so one summarize job yields both the overall number and
        # ``cpwer::<set>`` for each set.
        self.cpwer_subset_totals: dict[str, dict[str, int]] = {}

        self.wer_c_scores = []
        self.wer_pc_scores = []
        self.per_scores = []
        self.asr_pc_wer_totals = {
            metric_name: {
                "errors": 0,
                "ref_words": 0,
                "substitutions": 0,
                "insertions": 0,
                "deletions": 0,
            }
            for metric_name in ("wer_c", "wer_pc")
        }
        self.per_totals = {
            "correct": 0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "errors": 0,
            "operations": 0,
        }
        self.bleu_hyps: list[str] = []
        self.bleu_refs: list[str] = []
        self.bleu_tokenizes: list[str] = []
        self.bleu_subset_inputs: dict[str, dict[str, Any]] = {}

        self.comet_scores = []

        # Extended metrics
        self.cer_scores = []

        # Corpus-level CER accumulators (total errors / total ref chars)
        self.cer_total_errors = 0
        self.cer_total_ref_chars = 0
        self.cer_total_substitutions = 0
        self.cer_total_insertions = 0
        self.cer_total_deletions = 0
        self.cer_subset_totals: dict[str, dict[str, int]] = {}

        self.hallucination_scores = []
        self.pc_rate_scores = []
        self.punct_f1_scores = []
        self.cap_accuracy_scores = []
        self.total_hallucinated_chars = 0
        self.total_audio_seconds = 0.0

        # Judge scores (a 0-5 rating, or binary Yes/No mapped to 1/0)
        self.judge_ratings = []

        # Dataset-specific WER variants from additional reference fields (e.g., wer_tn, wer_itn).
        self.reference_wer_scores: dict[str, list[float]] = {}

        # Structured ASR scoring exposes strict/canonical/content WERs as separate streams.
        self.structured_wer_scores: dict[str, list[float]] = {}
        self.structured_wer_totals: dict[str, dict[str, int]] = {}
        self.structured_wer_subset_totals: dict[str, dict[str, dict[str, int]]] = {}

    def _extract_judge_result(self, judgement_text: str) -> tuple[bool, float]:
        """Extract judge result from judgement text.

        Supports two formats:
        1. Rating format: 'Rating: X' where X is 0-5 (returns rating as float)
        2. Binary format: 'Judgement: Yes/No' (mapped to 5.0/0.0 for consistent 0-100 scaling)

        Returns:
            Tuple of (is_correct, rating_score)
            - is_correct: True if rating >= 3 (or Yes for the binary format)
            - rating_score: 0-5 rating (or 0/5 for the binary format)
        """
        import re

        # Try the rating format first: 'Rating: X'
        rating_match = re.search(r"Rating:\s*([0-9]+(?:\.[0-9]+)?)", judgement_text, re.IGNORECASE)
        if rating_match:
            rating = float(rating_match.group(1))
            rating = max(0.0, min(5.0, rating))
            return rating >= 3.0, rating

        # Try explicit Judgement: Yes/No format
        judgement_match = re.search(r"Judgement:\s*(Yes|No)", judgement_text, re.IGNORECASE)
        if judgement_match:
            is_yes = judgement_match.group(1).lower() == "yes"
            return is_yes, 5.0 if is_yes else 0.0

        # Last-resort: accept plain 'yes'/'no' anywhere in text
        if re.search(r"\byes\b", judgement_text, re.IGNORECASE):
            return True, 5.0
        if re.search(r"\bno\b", judgement_text, re.IGNORECASE):
            return False, 0.0

        return False, 0.0

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Extract correctness scores from prediction.

        Handles both automatic metrics and judge-based evaluation,
        determining the overall correctness of a prediction.

        Args:
            prediction: Prediction dictionary with metrics and/or judgement

        Returns:
            Dictionary with correctness scores
        """
        score_dict = {}

        category = prediction.get("category", "unknown")

        if "judgement" in prediction and category == "open":
            judge_correct, judge_rating = self._extract_judge_result(prediction["judgement"])
            score_dict["judge_correct"] = judge_correct
            score_dict["judge_rating"] = judge_rating

        if category == "open" and "judge_correct" in score_dict:
            score_dict["correct"] = score_dict["judge_correct"]
        elif "is_correct" in prediction and prediction["is_correct"] is not None:
            score_dict["correct"] = prediction["is_correct"]
        else:
            score_dict["correct"] = False

        return score_dict

    def update_common_metrics(self, agg_dict):
        """Override to always include avg_tokens even if 0 since it's in metrics_to_print.

        Updates common metrics like number of entries, average tokens, and generation time.

        Args:
            agg_dict: Dictionary to update with common metrics
        """
        agg_dict["num_entries"] = self.total
        agg_dict["avg_tokens"] = int(self.avg_tokens / self.total) if self.total > 0 else 0
        if self.max_end_time > float("-inf") and self.min_start_time < float("inf"):
            agg_dict["gen_seconds"] = int(self.max_end_time - self.min_start_time)

    def update(self, predictions):
        """Update metrics with the given predictions.

        Collects all metric scores from predictions and updates internal tracking lists.
        Supports WER, BLEU, CER, hallucination and PC rate.

        Args:
            predictions: List of prediction dictionaries with computed metrics
        """
        super().update(predictions)

        predicted_answers = [pred.get("generation", "").strip() or None for pred in predictions]

        # Collect existing metrics: WER, PnC, and BLEU scores
        for pred in predictions:
            if "wer_errors" in pred and "wer_ref_words" in pred:
                self.wer_total_errors += pred["wer_errors"]
                self.wer_total_ref_words += pred["wer_ref_words"]
                self.wer_total_substitutions += pred["wer_substitutions"]
                self.wer_total_insertions += pred["wer_insertions"]
                self.wer_total_deletions += pred["wer_deletions"]
                if subset := pred.get("subset_for_metrics"):
                    totals = self.wer_subset_totals.setdefault(
                        str(subset),
                        {
                            "errors": 0,
                            "ref_words": 0,
                            "substitutions": 0,
                            "insertions": 0,
                            "deletions": 0,
                            "num_entries": 0,
                        },
                    )
                    for name in ("errors", "ref_words", "substitutions", "insertions", "deletions"):
                        totals[name] += pred[f"wer_{name}"]
                    totals["num_entries"] += 1
            if "wer" in pred and pred["wer"] is not None:
                self.wer_scores.append(pred["wer"])

            if "cpwer_errors" in pred and "cpwer_ref_words" in pred:
                self.cpwer_total_errors += pred["cpwer_errors"]
                self.cpwer_total_ref_words += pred["cpwer_ref_words"]
                self.cpwer_total_substitutions += pred["cpwer_substitutions"]
                self.cpwer_total_insertions += pred["cpwer_insertions"]
                self.cpwer_total_deletions += pred["cpwer_deletions"]

                # Bucket the same counts by the record's subset so a merged
                # benchmark keeps a per-set cpWER breakdown.
                subset = pred.get("subset_for_metrics")
                if subset:
                    totals = self.cpwer_subset_totals.setdefault(
                        str(subset),
                        {
                            "errors": 0,
                            "ref_words": 0,
                            "substitutions": 0,
                            "insertions": 0,
                            "deletions": 0,
                            "num_entries": 0,
                        },
                    )
                    totals["errors"] += pred["cpwer_errors"]
                    totals["ref_words"] += pred["cpwer_ref_words"]
                    totals["substitutions"] += pred["cpwer_substitutions"]
                    totals["insertions"] += pred["cpwer_insertions"]
                    totals["deletions"] += pred["cpwer_deletions"]
                    totals["num_entries"] += 1
            if "cpwer" in pred and pred["cpwer"] is not None:
                self.cpwer_scores.append(pred["cpwer"])

            for wer_name in ("strict_wer", "canonical_wer", "content_wer"):
                if wer_name in pred and pred[wer_name] is not None:
                    self.structured_wer_scores.setdefault(wer_name, []).append(pred[wer_name])

                errors_key = f"{wer_name}_errors"
                ref_words_key = f"{wer_name}_ref_words"
                if errors_key in pred and ref_words_key in pred:
                    totals = self.structured_wer_totals.setdefault(
                        wer_name,
                        {
                            "errors": 0,
                            "ref_words": 0,
                            "substitutions": 0,
                            "insertions": 0,
                            "deletions": 0,
                        },
                    )
                    totals["errors"] += pred.get(errors_key, 0)
                    totals["ref_words"] += pred.get(ref_words_key, 0)
                    totals["substitutions"] += pred.get(f"{wer_name}_substitutions", 0)
                    totals["insertions"] += pred.get(f"{wer_name}_insertions", 0)
                    totals["deletions"] += pred.get(f"{wer_name}_deletions", 0)
                    subset = pred.get("subset_for_metrics")
                    if subset:
                        subset_totals = self.structured_wer_subset_totals.setdefault(wer_name, {}).setdefault(
                            str(subset),
                            {
                                "errors": 0,
                                "ref_words": 0,
                                "substitutions": 0,
                                "insertions": 0,
                                "deletions": 0,
                                "num_entries": 0,
                            },
                        )
                        subset_totals["errors"] += pred.get(errors_key, 0)
                        subset_totals["ref_words"] += pred.get(ref_words_key, 0)
                        subset_totals["substitutions"] += pred.get(f"{wer_name}_substitutions", 0)
                        subset_totals["insertions"] += pred.get(f"{wer_name}_insertions", 0)
                        subset_totals["deletions"] += pred.get(f"{wer_name}_deletions", 0)
                        subset_totals["num_entries"] += 1
            if "wer_c" in pred and pred["wer_c"] is not None:
                self.wer_c_scores.append(pred["wer_c"])
            if "wer_pc" in pred and pred["wer_pc"] is not None:
                self.wer_pc_scores.append(pred["wer_pc"])
            if "per" in pred and pred["per"] is not None:
                self.per_scores.append(pred["per"])
            for metric_name in ("wer_c", "wer_pc"):
                if f"{metric_name}_errors" in pred and f"{metric_name}_ref_words" in pred:
                    totals = self.asr_pc_wer_totals[metric_name]
                    totals["errors"] += pred[f"{metric_name}_errors"]
                    totals["ref_words"] += pred[f"{metric_name}_ref_words"]
                    totals["substitutions"] += pred[f"{metric_name}_substitutions"]
                    totals["insertions"] += pred[f"{metric_name}_insertions"]
                    totals["deletions"] += pred[f"{metric_name}_deletions"]
            if "per_operations" in pred:
                self.per_totals["correct"] += pred["per_correct"]
                self.per_totals["substitutions"] += pred["per_substitutions"]
                self.per_totals["insertions"] += pred["per_insertions"]
                self.per_totals["deletions"] += pred["per_deletions"]
                self.per_totals["errors"] += pred["per_errors"]
                self.per_totals["operations"] += pred["per_operations"]
            if "bleu" in pred and pred["bleu"] is not None:
                self.bleu_hyps.append(pred["pred_text"])
                self.bleu_refs.append(pred["text"])
                self.bleu_tokenizes.append(pred["bleu_tokenize"])
                if subset := pred.get("subset_for_metrics"):
                    inputs = self.bleu_subset_inputs.setdefault(
                        str(subset),
                        {
                            "hyps": [],
                            "refs": [],
                            "tokenizes": [],
                            "num_entries": 0,
                        },
                    )
                    inputs["hyps"].append(pred["pred_text"])
                    inputs["refs"].append(pred["text"])
                    inputs["tokenizes"].append(pred["bleu_tokenize"])
                    inputs["num_entries"] += 1
            if "comet" in pred and pred["comet"] is not None:
                self.comet_scores.append(pred["comet"])

            # Collect extended metrics
            if "cer_errors" in pred and "cer_ref_words" in pred:
                self.cer_total_errors += pred["cer_errors"]
                self.cer_total_ref_chars += pred["cer_ref_words"]
                self.cer_total_substitutions += pred["cer_substitutions"]
                self.cer_total_insertions += pred["cer_insertions"]
                self.cer_total_deletions += pred["cer_deletions"]
                if subset := pred.get("subset_for_metrics"):
                    totals = self.cer_subset_totals.setdefault(
                        str(subset),
                        {
                            "errors": 0,
                            "ref_chars": 0,
                            "substitutions": 0,
                            "insertions": 0,
                            "deletions": 0,
                            "num_entries": 0,
                        },
                    )
                    totals["errors"] += pred["cer_errors"]
                    totals["ref_chars"] += pred["cer_ref_words"]
                    for name in ("substitutions", "insertions", "deletions"):
                        totals[name] += pred[f"cer_{name}"]
                    totals["num_entries"] += 1
            if "cer" in pred and pred["cer"] is not None:
                self.cer_scores.append(pred["cer"])
            if "hallucination_rate" in pred and pred["hallucination_rate"] is not None:
                self.hallucination_scores.append(pred["hallucination_rate"])
            if "pc_rate" in pred and pred["pc_rate"] is not None:
                self.pc_rate_scores.append(pred["pc_rate"])
            if "punct_f1" in pred and pred["punct_f1"] is not None:
                self.punct_f1_scores.append(pred["punct_f1"])
            if "cap_accuracy" in pred and pred["cap_accuracy"] is not None:
                self.cap_accuracy_scores.append(pred["cap_accuracy"])

            if pred.get("task_type") == "Hallucination":
                predicted_text = pred.get("predicted_answer") or pred.get("generation") or ""
                audio_duration = pred.get("audio_duration", 0.0)
                if audio_duration > 0:
                    self.total_hallucinated_chars += len(predicted_text.strip())
                    self.total_audio_seconds += audio_duration

            # Collect judge ratings (0-5) from judge datasets if available
            score_dict = self._get_score_dict(pred)
            if "judge_rating" in score_dict:
                self.judge_ratings.append(score_dict["judge_rating"])

            # Collect dataset-specific WER variants from any configured reference fields.
            _wer_count_fields = {
                "wer_c",
                "wer_pc",
                "wer_errors",
                "wer_ref_words",
                "wer_substitutions",
                "wer_insertions",
                "wer_deletions",
                "wer_c_errors",
                "wer_c_ref_words",
                "wer_c_substitutions",
                "wer_c_insertions",
                "wer_c_deletions",
                "wer_pc_errors",
                "wer_pc_ref_words",
                "wer_pc_substitutions",
                "wer_pc_insertions",
                "wer_pc_deletions",
            }
            for metric_name, metric_value in pred.items():
                if (
                    metric_name.startswith("wer_")
                    and metric_name not in _wer_count_fields
                    and metric_value is not None
                ):
                    self.reference_wer_scores.setdefault(metric_name, []).append(metric_value)

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        """Get computed metrics.

        Aggregates all collected metric scores and computes averages.
        Converts error rates and scores to percentages for reporting.

        Returns:
            Dictionary of aggregated metrics by evaluation mode
        """
        metrics_dict = super().get_metrics()

        for agg_metrics in metrics_dict.values():
            if "no_answer" in agg_metrics:
                # Divide by 2.0 to compensate for double-counting in base metrics #MEL
                agg_metrics["no_answer"] = agg_metrics["no_answer"] / 2.0

            # Set success_rate based on correct field
            if "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]
            elif "judge_correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["judge_correct"]

            # Add judge_score if rating outputs were used.
            # Formula: judge_score = mean(ratings) * 20 (converts 0-5 scale to 0-100)
            if self.judge_ratings:
                avg_rating = sum(self.judge_ratings) / len(self.judge_ratings)
                agg_metrics["judge_score"] = avg_rating * 20

            # Add existing metrics: WER, PnC, and BLEU if available (convert to percentages and round to 2 decimals)
            if self.wer_total_ref_words > 0:
                agg_metrics["substitutions"] = self.wer_total_substitutions
                agg_metrics["insertions"] = self.wer_total_insertions
                agg_metrics["deletions"] = self.wer_total_deletions
                agg_metrics["ref_words"] = self.wer_total_ref_words
                agg_metrics["wer"] = round(100.0 * self.wer_total_errors / self.wer_total_ref_words, 2)
            for subset, totals in sorted(self.wer_subset_totals.items()):
                if totals["ref_words"] > 0:
                    prefix = f"wer::{subset}"
                    agg_metrics[prefix] = round(100.0 * totals["errors"] / totals["ref_words"], 2)
                    agg_metrics[f"{prefix}_errors"] = totals["errors"]
                    agg_metrics[f"{prefix}_ref_words"] = totals["ref_words"]
                    agg_metrics[f"{prefix}_substitutions"] = totals["substitutions"]
                    agg_metrics[f"{prefix}_insertions"] = totals["insertions"]
                    agg_metrics[f"{prefix}_deletions"] = totals["deletions"]
                    agg_metrics[f"{prefix}_num_entries"] = totals["num_entries"]
            if self.wer_scores:
                agg_metrics["wer_macro"] = round(100.0 * sum(self.wer_scores) / len(self.wer_scores), 2)
            if self.cpwer_total_ref_words > 0:
                agg_metrics["cpwer_substitutions"] = self.cpwer_total_substitutions
                agg_metrics["cpwer_insertions"] = self.cpwer_total_insertions
                agg_metrics["cpwer_deletions"] = self.cpwer_total_deletions
                agg_metrics["cpwer_ref_words"] = self.cpwer_total_ref_words
                agg_metrics["cpwer_errors"] = self.cpwer_total_errors
                agg_metrics["cpwer"] = round(100.0 * self.cpwer_total_errors / self.cpwer_total_ref_words, 2)
            if self.cpwer_scores:
                agg_metrics["cpwer_macro"] = round(100.0 * sum(self.cpwer_scores) / len(self.cpwer_scores), 2)
            # Per-subset cpWER breakdown (micro cpWER within each subset_for_metrics).
            for subset, totals in sorted(self.cpwer_subset_totals.items()):
                if totals["ref_words"] > 0:
                    agg_metrics[f"cpwer::{subset}"] = round(100.0 * totals["errors"] / totals["ref_words"], 2)
                    agg_metrics[f"cpwer::{subset}_errors"] = totals["errors"]
                    agg_metrics[f"cpwer::{subset}_substitutions"] = totals["substitutions"]
                    agg_metrics[f"cpwer::{subset}_insertions"] = totals["insertions"]
                    agg_metrics[f"cpwer::{subset}_deletions"] = totals["deletions"]
                    agg_metrics[f"cpwer::{subset}_ref_words"] = totals["ref_words"]
                    agg_metrics[f"cpwer::{subset}_num_entries"] = totals["num_entries"]
            for wer_name, totals in self.structured_wer_totals.items():
                ref_words = totals.get("ref_words", 0)
                if ref_words > 0:
                    agg_metrics[wer_name] = round(100.0 * totals.get("errors", 0) / ref_words, 2)
                    agg_metrics[f"{wer_name}_errors"] = totals.get("errors", 0)
                    agg_metrics[f"{wer_name}_substitutions"] = totals.get("substitutions", 0)
                    agg_metrics[f"{wer_name}_insertions"] = totals.get("insertions", 0)
                    agg_metrics[f"{wer_name}_deletions"] = totals.get("deletions", 0)
                    agg_metrics[f"{wer_name}_ref_words"] = ref_words
            for wer_name, subsets in self.structured_wer_subset_totals.items():
                for subset, totals in sorted(subsets.items()):
                    ref_words = totals.get("ref_words", 0)
                    if ref_words > 0:
                        prefix = f"{wer_name}::{subset}"
                        agg_metrics[prefix] = round(100.0 * totals.get("errors", 0) / ref_words, 2)
                        agg_metrics[f"{prefix}_errors"] = totals.get("errors", 0)
                        agg_metrics[f"{prefix}_ref_words"] = ref_words
                        agg_metrics[f"{prefix}_substitutions"] = totals.get("substitutions", 0)
                        agg_metrics[f"{prefix}_insertions"] = totals.get("insertions", 0)
                        agg_metrics[f"{prefix}_deletions"] = totals.get("deletions", 0)
                        agg_metrics[f"{prefix}_num_entries"] = totals.get("num_entries", 0)
            for wer_name, values in self.structured_wer_scores.items():
                if values:
                    agg_metrics[f"{wer_name}_macro"] = round(100.0 * sum(values) / len(values), 2)
            for metric_name, scores in (("wer_c", self.wer_c_scores), ("wer_pc", self.wer_pc_scores)):
                totals = self.asr_pc_wer_totals[metric_name]
                if totals["ref_words"] > 0:
                    agg_metrics[metric_name] = round(100.0 * totals["errors"] / totals["ref_words"], 2)
                    agg_metrics[f"{metric_name}_errors"] = totals["errors"]
                    agg_metrics[f"{metric_name}_substitutions"] = totals["substitutions"]
                    agg_metrics[f"{metric_name}_insertions"] = totals["insertions"]
                    agg_metrics[f"{metric_name}_deletions"] = totals["deletions"]
                    agg_metrics[f"{metric_name}_ref_words"] = totals["ref_words"]
                elif scores:
                    agg_metrics[metric_name] = round(100.0 * sum(scores) / len(scores), 2)
                if scores:
                    agg_metrics[f"{metric_name}_macro"] = round(100.0 * sum(scores) / len(scores), 2)
            if self.per_totals["operations"] > 0:
                agg_metrics["per"] = round(100.0 * self.per_totals["errors"] / self.per_totals["operations"], 2)
                for count_name, value in self.per_totals.items():
                    agg_metrics[f"per_{count_name}"] = value
            elif self.per_scores:
                agg_metrics["per"] = round(100.0 * sum(self.per_scores) / len(self.per_scores), 2)
            if self.per_scores:
                agg_metrics["per_macro"] = round(100.0 * sum(self.per_scores) / len(self.per_scores), 2)
            if self.bleu_refs:
                agg_metrics["bleu"] = round(
                    compute_corpus_bleu(self.bleu_hyps, self.bleu_refs, self.bleu_tokenizes), 2
                )
            for subset, inputs in sorted(self.bleu_subset_inputs.items()):
                prefix = f"bleu::{subset}"
                agg_metrics[prefix] = round(
                    compute_corpus_bleu(inputs["hyps"], inputs["refs"], inputs["tokenizes"]),
                    2,
                )
                agg_metrics[f"{prefix}_num_entries"] = inputs["num_entries"]
            if self.comet_scores:
                agg_metrics["comet"] = round(100.0 * sum(self.comet_scores) / len(self.comet_scores), 2)

            # Add extended metrics if available
            if self.cer_total_ref_chars > 0:
                agg_metrics["cer_substitutions"] = self.cer_total_substitutions
                agg_metrics["cer_insertions"] = self.cer_total_insertions
                agg_metrics["cer_deletions"] = self.cer_total_deletions
                agg_metrics["cer_ref_chars"] = self.cer_total_ref_chars
                agg_metrics["cer"] = round(100.0 * self.cer_total_errors / self.cer_total_ref_chars, 2)
            for subset, totals in sorted(self.cer_subset_totals.items()):
                if totals["ref_chars"] > 0:
                    prefix = f"cer::{subset}"
                    agg_metrics[prefix] = round(100.0 * totals["errors"] / totals["ref_chars"], 2)
                    agg_metrics[f"{prefix}_errors"] = totals["errors"]
                    agg_metrics[f"{prefix}_ref_chars"] = totals["ref_chars"]
                    agg_metrics[f"{prefix}_substitutions"] = totals["substitutions"]
                    agg_metrics[f"{prefix}_insertions"] = totals["insertions"]
                    agg_metrics[f"{prefix}_deletions"] = totals["deletions"]
                    agg_metrics[f"{prefix}_num_entries"] = totals["num_entries"]
            if self.cer_scores:
                agg_metrics["cer_macro"] = round(100.0 * sum(self.cer_scores) / len(self.cer_scores), 2)
            if self.hallucination_scores:
                agg_metrics["hallucination_rate"] = round(
                    100.0 * sum(self.hallucination_scores) / len(self.hallucination_scores), 2
                )
            if self.pc_rate_scores:
                agg_metrics["pc_rate"] = round(100.0 * sum(self.pc_rate_scores) / len(self.pc_rate_scores), 2)
            if self.punct_f1_scores:
                agg_metrics["punct_f1"] = round(100.0 * sum(self.punct_f1_scores) / len(self.punct_f1_scores), 2)
            if self.cap_accuracy_scores:
                agg_metrics["cap_accuracy"] = round(
                    100.0 * sum(self.cap_accuracy_scores) / len(self.cap_accuracy_scores), 2
                )
            if self.total_audio_seconds > 0:
                total_minutes = self.total_audio_seconds / 60.0
                agg_metrics["char_rate"] = round(self.total_hallucinated_chars / total_minutes, 2)

            # Dataset-specific WER variants from additional reference fields.
            for metric_name, metric_values in self.reference_wer_scores.items():
                if metric_values:
                    agg_metrics[metric_name] = round(100.0 * sum(metric_values) / len(metric_values), 2)

        return metrics_dict
