# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge cases of the shared word edit-distance helpers."""

from resources_servers.audio_eval._scoring.edit_distance import edit_distance, word_error_counts


def test_word_error_counts_split_insertions_deletions_and_substitutions():
    assert word_error_counts(["a", "b", "c"], ["a", "c"]) == {
        "errors": 1,
        "substitutions": 0,
        "insertions": 0,
        "deletions": 1,
        "ref_words": 3,
    }
    assert word_error_counts([], ["x", "y"]) == {
        "errors": 2,
        "substitutions": 0,
        "insertions": 2,
        "deletions": 0,
        "ref_words": 0,
    }
    assert word_error_counts(["a", "b"], [])["deletions"] == 2
    assert word_error_counts(["a"], ["a"])["errors"] == 0


def test_edit_distance_is_symmetric_regardless_of_which_side_is_longer():
    assert edit_distance(["a"], ["a", "b", "c"]) == 2  # shorter reference swaps the operands internally
    assert edit_distance(["a", "b", "c"], ["a"]) == 2
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], []) == 0 and edit_distance([], ["x"]) == 1
