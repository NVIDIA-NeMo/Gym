# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA-NeMo/Skills (Apache-2.0), nemo_skills/evaluation/evaluator/contextual_earnings22.py
# at commit da85a881d972e6fec847b90cf553a0bf9bf10638: https://github.com/NVIDIA-NeMo/Skills
# Modifications: imports point at this repository's `audio_eval` scorer package and `nested_dataclass` is replaced by
# `dataclass`; the `open_asr` normalization mode
# (Open ASR English normalizer from `resources_servers/asr_with_pc`, applied to hypothesis, cased reference and
# keywords) is added to the sample function and the evaluator config; an unknown mode is rejected when the evaluator
# is constructed; the rest is unchanged.

"""Contextual Earnings-22 evaluator: WER + keyword Precision / Recall / F1.

Implements the metric defined in the Contextual Earnings-22 paper (Argmax,
2025, Section 3):

- WER: standard word error rate computed with edit-distance DP.
- Keyword TP/FP/FN: a keyword is a True Positive iff it matches the
  reference text AND its alignment position (computed via minimum edit
  distance) maps to identical hypothesis tokens. Otherwise it is an FN
  (in reference but not hypothesis at the aligned position) or an FP
  (in hypothesis but not reference at that position).

Per-sample raw counts are accumulated to corpus-level micro
precision / recall / F1 in ``ContextualEarnings22Metrics``.
"""

import re
from dataclasses import dataclass

from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig


NORMALIZATION_MODES = ("basic", "open_asr")

_OP_MATCH = "M"
_OP_SUB = "S"
_OP_INS = "I"
_OP_DEL = "D"


def normalize_text(text: str) -> str:
    """Mirror the ``norm_text`` field's normalization.

    Lowercases, replaces any non-letter/non-digit/non-whitespace character
    (and underscores) with a space, then collapses whitespace. Preserves
    Unicode letters via ``re.UNICODE``.

    Examples::

        "Q&A session"     -> "q a session"
        "Hyoung-il"       -> "hyoung il"
        "we'll"           -> "we ll"
        "100% of"         -> "100 of"
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


_OPEN_ASR_NORMALIZER = None


def normalize_open_asr(text: str) -> str:
    """Open ASR Leaderboard English normalizer (spoken numbers -> digits, spelling variants, etc.)."""
    global _OPEN_ASR_NORMALIZER
    if _OPEN_ASR_NORMALIZER is None:
        from resources_servers.asr_with_pc.open_asr_normalizer import (
            OpenASREnglishTextNormalizer,
        )

        _OPEN_ASR_NORMALIZER = OpenASREnglishTextNormalizer()
    return re.sub(r"\s+", " ", _OPEN_ASR_NORMALIZER(text)).strip()


def calculate_wer_with_alignment(hyp_tokens: list[str], ref_tokens: list[str]):
    """Word-level edit distance + traceback.

    Returns ``(wer, errors, ref_to_hyp)`` where:

    - ``wer``: insertions + deletions + substitutions, divided by ``len(ref_tokens)``
      (or by ``len(hyp_tokens)`` when the reference is empty, matching the paper's
      degenerate case behavior).
    - ``errors``: total raw error count.
    - ``ref_to_hyp``: list of length ``len(ref_tokens)``. Each entry is either
      ``(op, hyp_index)`` where ``op`` is ``"M"`` (match) or ``"S"``
      (substitution) and ``hyp_index`` is the 0-based hypothesis index aligned
      to that reference token; or ``("D", None)`` for a deletion (the reference
      token has no hypothesis counterpart).
    """
    n_hyp = len(hyp_tokens)
    n_ref = len(ref_tokens)

    if n_ref == 0:
        wer = 0.0 if n_hyp == 0 else float(n_hyp)
        return wer, n_hyp, []

    dp = [[0] * (n_ref + 1) for _ in range(n_hyp + 1)]
    bt = [[None] * (n_ref + 1) for _ in range(n_hyp + 1)]

    for i in range(1, n_hyp + 1):
        dp[i][0] = i
        bt[i][0] = _OP_INS
    for j in range(1, n_ref + 1):
        dp[0][j] = j
        bt[0][j] = _OP_DEL

    for i in range(1, n_hyp + 1):
        for j in range(1, n_ref + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = _OP_MATCH
                continue
            sub = dp[i - 1][j - 1] + 1
            ins = dp[i - 1][j] + 1
            dele = dp[i][j - 1] + 1
            best = min(sub, ins, dele)
            dp[i][j] = best
            if best == sub:
                bt[i][j] = _OP_SUB
            elif best == dele:
                bt[i][j] = _OP_DEL
            else:
                bt[i][j] = _OP_INS

    ref_to_hyp: list[tuple[str, int | None]] = [None] * n_ref  # type: ignore[list-item]
    i, j = n_hyp, n_ref
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == _OP_MATCH:
            ref_to_hyp[j - 1] = (_OP_MATCH, i - 1)
            i -= 1
            j -= 1
        elif op == _OP_SUB:
            ref_to_hyp[j - 1] = (_OP_SUB, i - 1)
            i -= 1
            j -= 1
        elif op == _OP_DEL:
            ref_to_hyp[j - 1] = (_OP_DEL, None)
            j -= 1
        elif op == _OP_INS:
            i -= 1
        else:
            break

    errors = dp[n_hyp][n_ref]
    wer = errors / n_ref
    return wer, errors, ref_to_hyp


def _find_subseq_occurrences(tokens: list[str], pattern: list[str]) -> list[int]:
    """Return all start indices of non-overlapping occurrences of ``pattern`` in ``tokens``.

    Greedy left-to-right scan; once a match is consumed, the next search
    resumes after its end. Returns an empty list if ``pattern`` is empty
    or longer than ``tokens``.
    """
    n, m = len(tokens), len(pattern)
    if m == 0 or m > n:
        return []
    out = []
    i = 0
    while i <= n - m:
        if tokens[i : i + m] == pattern:
            out.append(i)
            i += m
        else:
            i += 1
    return out


def evaluate_contextual_earnings22_sample(data_point: dict, normalization_mode: str = "basic") -> dict:
    """Evaluate a single Contextual Earnings-22 sample.

    ``normalization_mode="basic"`` (default) lowercases and strips punctuation on both sides and
    scores against the dataset ``norm_text`` (``expected_answer``). ``"open_asr"`` applies the Open
    ASR English normalizer to the hypothesis, the cased reference (``expected_answer_cased``) and the
    keywords (a model that writes "twenty twenty one" is otherwise charged errors against the digit
    reference "2021").

    Returns per-sample metrics with raw counts for corpus-level aggregation:
    ``wer``, ``wer_errors``, ``wer_ref_words``, ``keyword_tp``, ``keyword_fp``,
    ``keyword_fn``, ``keyword_total_ref``, ``is_correct`` (= ``wer < 0.5``),
    plus the normalized ``text`` / ``pred_text`` for inspection.
    """
    if normalization_mode == "open_asr":
        normalize = normalize_open_asr
        reference = data_point.get("expected_answer_cased") or data_point["expected_answer"]
    elif normalization_mode == "basic":
        normalize = normalize_text
        reference = data_point["expected_answer"]
    else:
        raise ValueError(f"Unknown normalization_mode {normalization_mode!r}; use 'basic' or 'open_asr'")
    generation = data_point["generation"].strip()
    keyword_list = data_point["keyword_list"]

    norm_ref = normalize(reference)
    norm_hyp = normalize(generation)

    ref_tokens = norm_ref.split()
    hyp_tokens = norm_hyp.split()

    wer, wer_errors, ref_to_hyp = calculate_wer_with_alignment(hyp_tokens, ref_tokens)
    wer_ref_words = len(ref_tokens)

    norm_keywords = []
    for kw in keyword_list:
        nk = normalize(kw)
        if nk:
            norm_keywords.append(nk.split())

    tp = 0
    fp = 0
    fn = 0
    total_ref_keyword_instances = 0

    matched_hyp_spans: set[tuple[int, int]] = set()

    for kw_tokens in norm_keywords:
        kw_len = len(kw_tokens)
        ref_starts = _find_subseq_occurrences(ref_tokens, kw_tokens)
        total_ref_keyword_instances += len(ref_starts)

        for ref_start in ref_starts:
            aligned_hyp = []
            exact = True
            for k in range(kw_len):
                op_info = ref_to_hyp[ref_start + k]
                if op_info is None:
                    exact = False
                    break
                op, hyp_idx = op_info
                if op != _OP_MATCH or hyp_idx is None:
                    exact = False
                    break
                aligned_hyp.append(hyp_idx)

            if exact and len(aligned_hyp) == kw_len:
                hyp_start = aligned_hyp[0]
                contiguous = all(
                    aligned_hyp[k] == hyp_start + k and hyp_tokens[hyp_start + k] == kw_tokens[k]
                    for k in range(kw_len)
                )
                if contiguous:
                    tp += 1
                    matched_hyp_spans.add((hyp_start, hyp_start + kw_len))
                    continue
            fn += 1

        hyp_starts = _find_subseq_occurrences(hyp_tokens, kw_tokens)
        hyp_unmatched = sum(1 for hs in hyp_starts if (hs, hs + kw_len) not in matched_hyp_spans)
        fp += hyp_unmatched

    return {
        "wer": wer,
        "wer_errors": wer_errors,
        "wer_ref_words": wer_ref_words,
        "keyword_tp": tp,
        "keyword_fp": fp,
        "keyword_fn": fn,
        "keyword_total_ref": total_ref_keyword_instances,
        "is_correct": wer < 0.5,
        "text": norm_ref,
        "pred_text": norm_hyp,
    }


@dataclass(kw_only=True)
class ContextualEarnings22EvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for Contextual Earnings-22 evaluation."""

    normalization_mode: str = "basic"


class ContextualEarnings22Evaluator(BaseEvaluator):
    """Evaluator for Contextual Earnings-22: WER + keyword P/R/F1."""

    def __init__(self, config: dict, num_parallel_requests: int = 10):
        """Initialize with evaluator config and parallelism settings."""
        super().__init__(config, num_parallel_requests)
        self.eval_config = ContextualEarnings22EvaluatorConfig(**config)
        if self.eval_config.normalization_mode not in NORMALIZATION_MODES:
            raise ValueError(
                f"Unknown normalization_mode {self.eval_config.normalization_mode!r}; "
                f"use one of {', '.join(NORMALIZATION_MODES)}"
            )

    async def eval_single(self, data_point: dict) -> dict:
        """Evaluate a single sample, returning WER + keyword TP/FP/FN counts."""
        return evaluate_contextual_earnings22_sample(data_point, self.eval_config.normalization_mode)
