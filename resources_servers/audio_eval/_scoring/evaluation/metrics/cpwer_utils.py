# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA-NeMo/NeMo (Apache-2.0), nemo/collections/asr/metrics/cpwer.py:
# https://github.com/NVIDIA-NeMo/NeMo at commit ffc18a65e4247e32720e4c59aaccda162fd00346 (2026-06-02, the first
# revision that computes edit distances with kaldialign, as this file does). The three functions are unchanged in
# NeMo's latest revision of that file, 1f73a910bc0f5ab774d100706329730b81457591 (2026-09-03, license-header change
# only). Modifications: type annotations, shortened docstrings and `session_cpwer_counts` / `_edit_counts` (error
# breakdown for corpus aggregation) added.

"""NeMo-compatible concatenated minimum-permutation WER (cpWER) utilities.

Adapted from ``nemo.collections.asr.metrics.cpwer`` (see the header for the revision) with a corpus-count
wrapper for corpus aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import permutations
from typing import Any

import numpy as np
from kaldialign import edit_distance
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment


__all__ = [
    "calculate_session_cpWER",
    "calculate_session_cpWER_bruteforce",
    "concat_perm_word_error_rate",
    "session_cpwer_counts",
]


def _edit_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    measures = edit_distance(reference, hypothesis)
    return {
        "errors": int(measures["total"]),
        "substitutions": int(measures["sub"]),
        "insertions": int(measures["ins"]),
        "deletions": int(measures["del"]),
        "ref_words": int(measures["ref_len"]),
    }


def calculate_session_cpWER_bruteforce(spk_hypothesis: list[str], spk_reference: list[str]) -> tuple[float, str, str]:
    """Calculate cpWER with brute-force permutation search."""
    num_hyp = len(spk_hypothesis)
    num_ref = len(spk_reference)
    num_speakers_padded = max(num_hyp, num_ref)

    ref_word_lists = [
        spk_reference[ref_idx].split() if ref_idx < num_ref else [] for ref_idx in range(num_speakers_padded)
    ]
    hyp_word_lists = [
        spk_hypothesis[hyp_idx].split() if hyp_idx < num_hyp else [] for hyp_idx in range(num_speakers_padded)
    ]

    best_total_errors = float("inf")
    best_hyp_trans = ""
    total_ref_length = sum(len(word_list) for word_list in ref_word_lists)

    for perm in permutations(range(num_speakers_padded)):
        total_errors = 0
        hyp_texts = []
        for ref_idx, hyp_idx in enumerate(perm):
            total_errors += edit_distance(ref_word_lists[ref_idx], hyp_word_lists[hyp_idx])["total"]
            hyp_texts.append(spk_hypothesis[hyp_idx] if hyp_idx < num_hyp else "")
        if total_errors < best_total_errors:
            best_total_errors = total_errors
            best_hyp_trans = " ".join(hyp_texts)

    cpwer = best_total_errors / total_ref_length if total_ref_length > 0 else float("inf")
    ref_trans = " ".join(spk_reference)
    return cpwer, best_hyp_trans, ref_trans


def calculate_session_cpWER(spk_hypothesis: list[str], spk_reference: list[str]) -> tuple[float, str, str]:
    """Calculate session-level cpWER using Hungarian speaker assignment."""
    num_hyp = len(spk_hypothesis)
    num_ref = len(spk_reference)

    if num_hyp == 0 and num_ref == 0:
        return 0.0, "", ""

    num_speakers_padded = max(num_hyp, num_ref)

    ref_word_lists = [
        spk_reference[ref_idx].split() if ref_idx < num_ref else [] for ref_idx in range(num_speakers_padded)
    ]
    hyp_word_lists = [
        spk_hypothesis[hyp_idx].split() if hyp_idx < num_hyp else [] for hyp_idx in range(num_speakers_padded)
    ]

    cost_matrix = np.zeros((num_speakers_padded, num_speakers_padded), dtype=np.float64)
    for ref_idx in range(num_speakers_padded):
        for hyp_idx in range(num_speakers_padded):
            cost_matrix[ref_idx, hyp_idx] = edit_distance(ref_word_lists[ref_idx], hyp_word_lists[hyp_idx])["total"]

    row_ind, col_ind = scipy_linear_sum_assignment(cost_matrix)

    total_errors = 0
    total_ref_length = 0
    hyp_texts = []
    for ref_idx, hyp_idx in zip(row_ind, col_ind):
        total_errors += int(cost_matrix[ref_idx, hyp_idx])
        total_ref_length += len(ref_word_lists[ref_idx])
        hyp_texts.append(spk_hypothesis[hyp_idx] if hyp_idx < num_hyp else "")

    cpwer = total_errors / total_ref_length if total_ref_length > 0 else float("inf")

    min_perm_hyp_trans = " ".join(hyp_texts)
    ref_trans = " ".join(spk_reference)

    return cpwer, min_perm_hyp_trans, ref_trans


def session_cpwer_counts(
    spk_reference: Sequence[str] | Mapping[str, str],
    spk_hypothesis: Sequence[str] | Mapping[str, str],
) -> dict[str, Any]:
    """Compute cpWER and detailed error counts for one session."""
    ref_streams = list(spk_reference.values()) if isinstance(spk_reference, Mapping) else list(spk_reference)
    hyp_streams = list(spk_hypothesis.values()) if isinstance(spk_hypothesis, Mapping) else list(spk_hypothesis)

    ref_word_lists = [stream.split() for stream in ref_streams]
    hyp_word_lists = [stream.split() for stream in hyp_streams]
    num_ref = len(ref_word_lists)
    num_hyp = len(hyp_word_lists)

    if num_hyp == 0 and num_ref == 0:
        return {
            "cpwer": 0.0,
            "errors": 0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "ref_words": 0,
            "assignment": [],
        }

    num_speakers_padded = max(num_hyp, num_ref)
    while len(ref_word_lists) < num_speakers_padded:
        ref_word_lists.append([])
    while len(hyp_word_lists) < num_speakers_padded:
        hyp_word_lists.append([])

    cost_matrix = np.zeros((num_speakers_padded, num_speakers_padded), dtype=np.float64)
    pair_counts: list[list[dict[str, int]]] = [
        [{} for _ in range(num_speakers_padded)] for _ in range(num_speakers_padded)
    ]
    for ref_idx in range(num_speakers_padded):
        for hyp_idx in range(num_speakers_padded):
            counts = _edit_counts(ref_word_lists[ref_idx], hyp_word_lists[hyp_idx])
            pair_counts[ref_idx][hyp_idx] = counts
            cost_matrix[ref_idx, hyp_idx] = counts["errors"]

    row_ind, col_ind = scipy_linear_sum_assignment(cost_matrix)

    totals = {"errors": 0, "substitutions": 0, "insertions": 0, "deletions": 0, "ref_words": 0}
    assignment = []
    for ref_idx, hyp_idx in zip(row_ind, col_ind):
        counts = pair_counts[ref_idx][hyp_idx]
        for key in totals:
            totals[key] += counts[key]
        assignment.append(
            {
                "reference": f"s{ref_idx}" if ref_idx < num_ref else f"__empty_ref_{ref_idx}",
                "hypothesis": f"s{hyp_idx}" if hyp_idx < num_hyp else f"__empty_hyp_{hyp_idx}",
            }
        )

    ref_words = totals["ref_words"]
    cpwer = totals["errors"] / ref_words if ref_words > 0 else 0.0
    return {
        "cpwer": cpwer,
        "errors": totals["errors"],
        "substitutions": totals["substitutions"],
        "insertions": totals["insertions"],
        "deletions": totals["deletions"],
        "ref_words": ref_words,
        "assignment": assignment,
    }


def concat_perm_word_error_rate(
    spk_hypotheses: list[list[str]], spk_references: list[list[str]]
) -> tuple[list[float], list[str], list[str]]:
    """Launcher for ``calculate_session_cpWER`` over multiple sessions."""
    if len(spk_hypotheses) != len(spk_references):
        raise ValueError(
            "In concatenated-minimum permutation word error rate calculation, "
            "hypotheses and reference lists must have the same number of elements. But got arguments:"
            f"{len(spk_hypotheses)} and {len(spk_references)} correspondingly"
        )
    cpwer_values, hyps_spk, refs_spk = [], [], []
    for spk_hypothesis, spk_reference in zip(spk_hypotheses, spk_references):
        cpwer, min_hypothesis, concat_reference = calculate_session_cpWER(spk_hypothesis, spk_reference)
        cpwer_values.append(cpwer)
        hyps_spk.append(min_hypothesis)
        refs_spk.append(concat_reference)
    return cpwer_values, hyps_spk, refs_spk
