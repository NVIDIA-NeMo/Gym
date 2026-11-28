# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Corpus-micro standard and preference-aware WER for public Preference-ASR."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from resources_servers.audio_eval._scoring.edit_distance import word_error_counts
from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator
from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics

from .normalizer import preference_normalizer, standard_normalizer


PREFERENCE_TYPES = ("normalization", "entities", "disfluencies", "case", "standard")
SOURCE_DATASETS = ("ami", "librispeech", "common_voice", "voxpopuli")


def _normalized_counts(reference: str, hypothesis: str, normalizer) -> tuple[str, str, dict[str, int]]:
    normalized_reference = normalizer(reference)
    normalized_hypothesis = normalizer(hypothesis)
    return (
        normalized_reference,
        normalized_hypothesis,
        word_error_counts(normalized_reference.split(), normalized_hypothesis.split()),
    )


def evaluate_preference_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    generation = str(data_point.get("generation") or "").strip()
    category = data_point.get("preference_type")
    source_dataset = data_point.get("preference_source_dataset")
    prompt_variant = data_point.get("preference_prompt_variant")
    if category not in PREFERENCE_TYPES:
        raise ValueError(f"Unsupported Preference-ASR category: {category!r}")
    if source_dataset not in SOURCE_DATASETS:
        raise ValueError(f"Unsupported Preference-ASR source dataset: {source_dataset!r}")
    if prompt_variant not in {"default", "instructed"}:
        raise ValueError(f"Unsupported Preference-ASR prompt variant: {prompt_variant!r}")
    reference = str(data_point["expected_answer"])
    std_ref, std_hyp, std_counts = _normalized_counts(reference, generation, standard_normalizer())
    pref = preference_normalizer()
    pref_ref, pref_hyp, pref_counts = _normalized_counts(
        reference,
        generation,
        lambda text: pref.normalize_entry(text, data_point),
    )
    values: dict[str, Any] = {
        "generation": generation,
        "text": reference,
        "pred_text": generation,
        "preference_type": category,
        "preference_source_dataset": source_dataset,
        "preference_prompt_variant": prompt_variant,
        "standard_normalized_reference": std_ref,
        "standard_normalized_hypothesis": std_hyp,
        "preference_normalized_reference": pref_ref,
        "preference_normalized_hypothesis": pref_hyp,
        "missing_prediction": not bool(generation),
        "is_correct": pref_counts["errors"] == 0,
    }
    values.update({f"standard_{name}": value for name, value in std_counts.items()})
    values.update({f"preference_{name}": value for name, value in pref_counts.items()})
    return values


class PreferenceASREvaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_preference_sample(data_point)


class PreferenceASRMetrics(BaseMetrics):
    """Official table dimensions, with raw corpus-micro count evidence."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("PreferenceASRMetrics supports one greedy generation only")
        self.max_k = 1

    def reset(self):
        super().reset()
        self.aggregates = {metric: defaultdict(lambda: defaultdict(int)) for metric in ("standard", "preference")}
        self.prompt_variant: str | None = None
        self.missing_predictions = 0

    def _get_score_dict(self, prediction):
        return {"preference_exact": int(prediction.get("preference_errors", 1)) == 0}

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("PreferenceASRMetrics expects exactly one prediction per sample")
        prediction = predictions[0]
        variant = prediction["preference_prompt_variant"]
        if self.prompt_variant is not None and variant != self.prompt_variant:
            raise ValueError("Default and instructed Preference-ASR rows must be summarized separately")
        self.prompt_variant = variant
        super().update(predictions)
        self.missing_predictions += int(bool(prediction["missing_prediction"]))
        keys = (
            "overall",
            f"type:{prediction['preference_type']}",
            f"dataset:{prediction['preference_source_dataset']}",
        )
        for metric in ("standard", "preference"):
            for key in keys:
                target = self.aggregates[metric][key]
                for count in ("errors", "substitutions", "insertions", "deletions", "ref_words"):
                    target[count] += int(prediction[f"{metric}_{count}"])
                target["entries"] += 1

    @staticmethod
    def _wer(counts: dict[str, int]) -> float:
        if counts["ref_words"] <= 0:
            raise ValueError("Preference-ASR aggregate has no reference words")
        return round(100.0 * counts["errors"] / counts["ref_words"], 2)

    def get_metrics(self):
        if self.total <= 0 or self.prompt_variant is None:
            raise ValueError("Preference-ASR has no predictions to score")
        values: dict[str, Any] = {
            "prompt_variant": self.prompt_variant,
            "coverage": round(100.0 * (self.total - self.missing_predictions) / self.total, 2),
            "missing_predictions": self.missing_predictions,
            "scored_entries": self.total,
        }
        for metric in ("standard", "preference"):
            overall = self.aggregates[metric]["overall"]
            values[f"{metric}_wer"] = self._wer(overall)
            for count in ("errors", "substitutions", "insertions", "deletions", "ref_words"):
                values[f"{metric}_{count}"] = overall[count]
            for category in PREFERENCE_TYPES:
                counts = self.aggregates[metric].get(f"type:{category}")
                if counts:
                    values[f"{metric}_wer_{category}"] = self._wer(counts)
                    values[f"entries_{category}"] = counts["entries"]
            for source_dataset in SOURCE_DATASETS:
                counts = self.aggregates[metric].get(f"dataset:{source_dataset}")
                if counts:
                    values[f"{metric}_wer_{source_dataset}"] = self._wer(counts)
        self.update_common_metrics(values)
        return {"pass@1": values}
