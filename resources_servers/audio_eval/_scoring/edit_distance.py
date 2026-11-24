# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Word-level edit distance helpers shared by the dataset scorers."""


def word_error_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    """Deterministic Levenshtein substitution/insertion/deletion counts (fixed backtrace tie ordering)."""
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    distances = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for ref_index in range(ref_len + 1):
        distances[ref_index][0] = ref_index
    for hyp_index in range(hyp_len + 1):
        distances[0][hyp_index] = hyp_index
    for ref_index in range(1, ref_len + 1):
        for hyp_index in range(1, hyp_len + 1):
            if reference[ref_index - 1] == hypothesis[hyp_index - 1]:
                distances[ref_index][hyp_index] = distances[ref_index - 1][hyp_index - 1]
            else:
                distances[ref_index][hyp_index] = 1 + min(
                    distances[ref_index - 1][hyp_index - 1],
                    distances[ref_index][hyp_index - 1],
                    distances[ref_index - 1][hyp_index],
                )

    substitutions = insertions = deletions = 0
    ref_index, hyp_index = ref_len, hyp_len
    while ref_index > 0 or hyp_index > 0:
        if ref_index > 0 and hyp_index > 0 and reference[ref_index - 1] == hypothesis[hyp_index - 1]:
            ref_index -= 1
            hyp_index -= 1
        elif (
            ref_index > 0
            and hyp_index > 0
            and distances[ref_index][hyp_index] == distances[ref_index - 1][hyp_index - 1] + 1
        ):
            substitutions += 1
            ref_index -= 1
            hyp_index -= 1
        elif hyp_index > 0 and distances[ref_index][hyp_index] == distances[ref_index][hyp_index - 1] + 1:
            insertions += 1
            hyp_index -= 1
        elif ref_index > 0 and distances[ref_index][hyp_index] == distances[ref_index - 1][hyp_index] + 1:
            deletions += 1
            ref_index -= 1
        else:
            raise RuntimeError("Word edit-distance backtrace reached an inconsistent state")

    errors = substitutions + insertions + deletions
    if errors != distances[ref_len][hyp_len]:
        raise RuntimeError("Word edit-distance error-count invariant failed")
    return {
        "errors": errors,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "ref_words": ref_len,
    }


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance with O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        hypothesis, reference = reference, hypothesis
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]
