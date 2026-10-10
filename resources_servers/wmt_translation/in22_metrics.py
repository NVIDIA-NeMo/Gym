# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IN22 response cleanup, Indic normalization, and reference-compatible metrics."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Any

from sacrebleu import corpus_bleu, corpus_chrf, sentence_bleu, sentence_chrf


INDIC_LANGUAGE_CODES = frozenset({"as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur"})


def clean_in22_response(text: str) -> str:
    """Remove reasoning and an optional Markdown fence exactly as the reference filter does."""
    text = str(text).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    elif "<think>" in text:
        return ""

    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


@lru_cache(maxsize=None)
def _indic_tools(language: str):
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import indic_tokenize

    try:
        normalizer = IndicNormalizerFactory().get_normalizer(language, remove_nuktas=False)
    except (ImportError, ModuleNotFoundError):
        normalizer = None
    return normalizer, indic_tokenize


def normalize_in22(text: str, language: str) -> str:
    """Apply the reference normalization/tokenization for one target language."""
    text = "" if text is None else str(text).strip()
    if language == "en":
        return text
    if language not in INDIC_LANGUAGE_CODES:
        raise ValueError(f"Unsupported IN22 target language: {language}")
    normalizer, indic_tokenize = _indic_tools(language)
    if normalizer is not None:
        text = normalizer.normalize(text)
    return " ".join(indic_tokenize.trivial_tokenize(text, language))


def sentence_metrics(reference: str, hypothesis: str, target_language: str) -> tuple[float, float, float]:
    """Return sentence chrF, chrF++, and BLEU for Gym's dense rollout reward."""
    reference = normalize_in22(reference, target_language)
    hypothesis = normalize_in22(hypothesis, target_language)
    tokenizer = "13a" if target_language == "en" else "none"
    return (
        sentence_chrf(hypothesis, [reference], word_order=0).score,
        sentence_chrf(hypothesis, [reference], word_order=2).score,
        sentence_bleu(hypothesis, [reference], tokenize=tokenizer, use_effective_order=True).score,
    )


def corpus_metrics(references: list[str], hypotheses: list[str], target_language: str) -> dict[str, float]:
    """Return the three corpus metrics configured by the reference lm-eval tasks."""
    references = [normalize_in22(text, target_language) for text in references]
    hypotheses = [normalize_in22(text, target_language) for text in hypotheses]
    tokenizer = "13a" if target_language == "en" else "none"
    return {
        "chrf": corpus_chrf(hypotheses, [references], word_order=0).score,
        "chrf++": corpus_chrf(hypotheses, [references], word_order=2).score,
        "bleu": corpus_bleu(hypotheses, [references], tokenize=tokenizer).score,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance**0.5


def compute_in22_metrics(tasks: list[list[dict[str, Any]]]) -> dict[str, float]:
    """Aggregate reference metrics by direction/language and macro-average them."""
    if not tasks:
        return {}
    num_runs = min((len(rollouts) for rollouts in tasks), default=0)
    per_pair_runs: dict[tuple[str, str], list[list[tuple[str, str]]]] = defaultdict(
        lambda: [list() for _ in range(num_runs)]
    )
    for rollouts in tasks:
        for run_index, rollout in enumerate(rollouts[:num_runs]):
            source = rollout.get("source_language")
            target = rollout.get("target_language")
            if not source or not target:
                continue
            per_pair_runs[(source, target)][run_index].append(
                (str(rollout.get("translation") or ""), str(rollout.get("generation") or ""))
            )

    per_pair: dict[tuple[str, str], dict[str, list[float]]] = {}
    for (source, target), runs in per_pair_runs.items():
        scores = {metric: [] for metric in ("chrf", "chrf++", "bleu")}
        for run in runs:
            if not run:
                continue
            references, hypotheses = zip(*run, strict=True)
            run_scores = corpus_metrics(list(references), list(hypotheses), target)
            for metric, score in run_scores.items():
                scores[metric].append(score)
        per_pair[(source, target)] = scores

    metrics: dict[str, float] = {}
    pairs = sorted(per_pair)
    for source, target in pairs:
        for metric, values in per_pair[(source, target)].items():
            mean, std = _mean_std(values)
            metrics[f"{source}->{target}/{metric}"] = mean
            metrics[f"{source}->{target}/{metric}_std_dev_across_runs"] = std

    def add_macro(label: str, selected: list[tuple[str, str]]) -> None:
        if not selected:
            return
        for metric in ("chrf", "chrf++", "bleu"):
            run_count = min((len(per_pair[pair][metric]) for pair in selected), default=0)
            values = [
                sum(per_pair[pair][metric][index] for pair in selected) / len(selected) for index in range(run_count)
            ]
            mean, std = _mean_std(values)
            metrics[f"{label}/{metric}"] = mean
            metrics[f"{label}/{metric}_std_dev_across_runs"] = std

    add_macro("xx->xx", pairs)
    for source in sorted({pair[0] for pair in pairs}):
        add_macro(f"{source}->xx", [pair for pair in pairs if pair[0] == source])
    for target in sorted({pair[1] for pair in pairs}):
        add_macro(f"xx->{target}", [pair for pair in pairs if pair[1] == target])
    return metrics
