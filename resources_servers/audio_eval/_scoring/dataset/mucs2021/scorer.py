# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol-faithful MUCS 2021 Subtask 2 blind-test scorer.

This is a direct, dependency-free implementation of the released
``inference/compute_wer.py`` protocol. Both references and hypotheses are
lowercased and whitespace-tokenized. The punctuation token map is applied to
hypotheses only for Normal WER. The language word map is then applied to those
already punctuation-mapped hypotheses only for Transliterated WER. Empty
references are skipped exactly as in the release.
"""

from __future__ import annotations

from typing import Any

from resources_servers.audio_eval._scoring.edit_distance import edit_distance
from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator
from resources_servers.audio_eval._scoring.evaluation.metrics.base import BaseMetrics
from resources_servers.audio_eval.mucs_assets import load_official_mapping


WORD_MAPPING_ASSET = {
    "hindi-english": "hin2engwords.txt",
    "bengali-english": "ben2engwords.txt",
}


def whitespace_tokens(text: object) -> list[str]:
    """Match ``line.strip().lower().split()`` from the official script."""
    return str(text).strip().lower().split()


def apply_mapping(tokens: list[str], mapping: dict[str, str]) -> list[str]:
    """Match the official join-after-token-replacement operation exactly."""
    return " ".join(mapping.get(token, token) for token in tokens).split()


def evaluate_mucs2021_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one row while retaining additive corpus counts."""
    language = str(data_point.get("extra_fields", {}).get("mucs_language", ""))
    if language not in WORD_MAPPING_ASSET:
        raise ValueError(f"Missing or invalid extra_fields.mucs_language: {language!r}")

    reference = whitespace_tokens(data_point["expected_answer"])
    generation = str(data_point.get("generation") or "").strip()
    hypothesis = whitespace_tokens(generation)
    punctuated_hypothesis = apply_mapping(hypothesis, load_official_mapping("punct.mapping.txt"))
    transliterated_hypothesis = apply_mapping(
        punctuated_hypothesis,
        load_official_mapping(WORD_MAPPING_ASSET[language]),
    )

    # The official loop continues before calling jiwer when the reference is
    # empty. Hypothesis insertions therefore do not contribute to either WER.
    if not reference:
        normal_errors = 0
        transliterated_errors = 0
        skipped = True
    else:
        normal_errors = edit_distance(reference, punctuated_hypothesis)
        transliterated_errors = edit_distance(reference, transliterated_hypothesis)
        skipped = False

    ref_words = len(reference)
    return {
        "generation": generation,
        "text": " ".join(reference),
        "pred_text": " ".join(punctuated_hypothesis),
        "transliterated_pred_text": " ".join(transliterated_hypothesis),
        "wer": normal_errors / ref_words if ref_words else None,
        "twer": transliterated_errors / ref_words if ref_words else None,
        "wer_errors": normal_errors,
        "twer_errors": transliterated_errors,
        "ref_words": ref_words,
        "skipped_empty_reference": skipped,
    }


class MUCS2021Evaluator(BaseEvaluator):
    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_mucs2021_sample(data_point)


class MUCS2021Metrics(BaseMetrics):
    """Corpus WER/tWER matching the official per-utterance weighted sum."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        if max_k != 1:
            raise ValueError("MUCS2021Metrics supports one greedy generation only")
        self.max_k = 1

    def reset(self):
        super().reset()
        self.wer_errors = 0
        self.twer_errors = 0
        self.ref_words = 0
        self.scored_entries = 0
        self.skipped_empty_references = 0
        self.no_answer = 0

    def _get_score_dict(self, prediction):
        return {"exact": prediction.get("wer_errors") == 0 and not prediction.get("skipped_empty_reference")}

    def update(self, predictions):
        if len(predictions) != 1:
            raise ValueError("MUCS2021Metrics expects exactly one prediction per sample")
        super().update(predictions)
        prediction = predictions[0]
        self.wer_errors += int(prediction["wer_errors"])
        self.twer_errors += int(prediction["twer_errors"])
        self.ref_words += int(prediction["ref_words"])
        if prediction["skipped_empty_reference"]:
            self.skipped_empty_references += 1
        else:
            self.scored_entries += 1
        if not (prediction.get("generation") or "").strip():
            self.no_answer += 1

    def get_metrics(self):
        if self.ref_words <= 0:
            raise ValueError("MUCS 2021 has no non-empty reference words to score")
        values = {
            "wer": round(100.0 * self.wer_errors / self.ref_words, 2),
            "twer": round(100.0 * self.twer_errors / self.ref_words, 2),
            "wer_errors": self.wer_errors,
            "twer_errors": self.twer_errors,
            "ref_words": self.ref_words,
            "scored_entries": self.scored_entries,
            "skipped_empty_references": self.skipped_empty_references,
            "no_answer": 100.0 * self.no_answer / self.total,
        }
        self.update_common_metrics(values)
        return {"pass@1": values}
