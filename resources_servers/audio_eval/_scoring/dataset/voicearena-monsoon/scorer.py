# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official aggregate WER/OIWER scoring for public Monsoon ASR splits."""

from __future__ import annotations

from functools import cache
from importlib.metadata import version
from typing import Any

from resources_servers.asr_with_pc.open_asr_normalizer import OpenASREnglishTextNormalizer
from resources_servers.audio_eval._scoring.edit_distance import word_error_counts
from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator
from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics


OIWER_DISTRIBUTION = "voi-oiwer"
OIWER_VERSION = "0.1.4"


@cache
def _english_normalizer() -> OpenASREnglishTextNormalizer:
    return OpenASREnglishTextNormalizer()


@cache
def _official_oiwer():
    actual_version = version(OIWER_DISTRIBUTION)
    if actual_version != OIWER_VERSION:
        raise RuntimeError(f"Monsoon hi-IN requires {OIWER_DISTRIBUTION}=={OIWER_VERSION}; found {actual_version}")
    from voi_oiwer import oiwer

    return oiwer


def _evaluate_english(data_point: dict[str, Any], generation: str) -> dict[str, Any]:
    reference = _english_normalizer()(str(data_point["expected_answer"]))
    hypothesis = _english_normalizer()(generation)
    reference_words = reference.split()
    if not reference_words:
        raise ValueError("Monsoon en-IN reference normalized to an empty string")
    counts = word_error_counts(reference_words, hypothesis.split())
    return {
        "monsoon_metric": "wer",
        "text": reference,
        "pred_text": hypothesis,
        **{f"monsoon_{name}": value for name, value in counts.items()},
    }


def _evaluate_hindi(data_point: dict[str, Any], generation: str) -> dict[str, Any]:
    lattice = data_point.get("reference_lattice")
    if not isinstance(lattice, list) or not lattice or any(not isinstance(slot, list) or not slot for slot in lattice):
        raise ValueError("Monsoon hi-IN requires a nonempty list-of-lists reference_lattice")
    result = _official_oiwer()(generation, lattice, "hindi")
    score, counts, ref_words = float(result[0]), result[4], int(result[5])
    if len(counts) != 3 or ref_words <= 0:
        raise ValueError("voi-oiwer returned an invalid count/reference contract")
    insertions, deletions, substitutions = (int(value) for value in counts)
    errors = insertions + deletions + substitutions
    if abs(score - 100.0 * errors / ref_words) > 1e-6:
        raise ValueError("voi-oiwer score/count invariant failed")
    return {
        "monsoon_metric": "oiwer",
        "text": data_point.get("expected_answer", ""),
        "pred_text": generation,
        "monsoon_errors": errors,
        "monsoon_substitutions": substitutions,
        "monsoon_insertions": insertions,
        "monsoon_deletions": deletions,
        "monsoon_ref_words": ref_words,
        "oiwer_utterance_diagnostic": score / 100.0,
    }


def evaluate_monsoon_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    generation = str(data_point.get("generation") or "").strip()
    language = data_point.get("monsoon_language")
    if language == "en-in":
        values = _evaluate_english(data_point, generation)
    elif language == "hi-in":
        values = _evaluate_hindi(data_point, generation)
    else:
        raise ValueError(f"Unsupported Monsoon language: {language!r}")
    values.update(
        generation=generation,
        missing_prediction=not bool(generation),
        is_correct=values["monsoon_errors"] == 0,
    )
    return values


class MonsoonASREvaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_monsoon_sample(data_point)


class MonsoonASRMetrics(BaseMetrics):
    """Corpus-micro official WER or OIWER with raw count evidence."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("MonsoonASRMetrics supports one greedy generation only")
        self.max_k = 1

    def reset(self):
        super().reset()
        self.metric_name: str | None = None
        self.errors = 0
        self.ref_words = 0
        self.substitutions = 0
        self.insertions = 0
        self.deletions = 0
        self.missing_predictions = 0

    def _get_score_dict(self, prediction):
        return {"exact": int(prediction.get("monsoon_errors", 1)) == 0}

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("MonsoonASRMetrics expects exactly one prediction per sample")
        prediction = predictions[0]
        metric_name = str(prediction["monsoon_metric"])
        if metric_name not in {"wer", "oiwer"}:
            raise ValueError(f"Unsupported Monsoon metric: {metric_name!r}")
        if self.metric_name is not None and metric_name != self.metric_name:
            raise ValueError("Monsoon WER and OIWER rows must be summarized as separate profiles")
        self.metric_name = metric_name
        super().update(predictions)
        self.errors += int(prediction["monsoon_errors"])
        self.ref_words += int(prediction["monsoon_ref_words"])
        self.substitutions += int(prediction["monsoon_substitutions"])
        self.insertions += int(prediction["monsoon_insertions"])
        self.deletions += int(prediction["monsoon_deletions"])
        self.missing_predictions += int(bool(prediction["missing_prediction"]))

    def get_metrics(self):
        if self.total <= 0 or self.ref_words <= 0 or self.metric_name is None:
            raise ValueError("Monsoon ASR has no predictions/reference words to score")
        values = {
            self.metric_name: round(100.0 * self.errors / self.ref_words, 2),
            "coverage": round(100.0 * (self.total - self.missing_predictions) / self.total, 2),
            "missing_predictions": self.missing_predictions,
            "scored_entries": self.total,
            "errors": self.errors,
            "ref_words": self.ref_words,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
        }
        self.update_common_metrics(values)
        return {"pass@1": values}
