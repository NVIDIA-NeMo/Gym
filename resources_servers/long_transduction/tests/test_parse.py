# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for long_transduction parsers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

from resources_servers.long_transduction import parse
from resources_servers.long_transduction.parse import (
    _eval_expr,
    _normalize_expr,
    _normalize_value,
    _parse_csv_grid,
    _parse_line,
    _parse_numbered_line,
    _parse_numbered_uuid_line,
    _parse_numbered_value_line,
    score_csv_permutation,
    score_response,
    score_response_numbered,
    score_unnumbered_uuid_sort,
    score_uuid_sort,
    score_var_expand_numbered,
    score_var_expand_unnumbered,
)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizeExpr:
    def test_strips_internal_and_edge_whitespace(self):
        assert _normalize_expr("5 + 6 - 3") == "5+6-3"
        assert _normalize_expr("  5+6  ") == "5+6"

    def test_already_normalized(self):
        assert _normalize_expr("5+6") == "5+6"


class TestParseLine:
    def test_basic(self):
        assert _parse_line("5+6=11") == ("5+6", 11.0)

    def test_with_whitespace(self):
        assert _parse_line(" 5 + 6 = 11 ") == ("5+6", 11.0)

    def test_negative_answer(self):
        assert _parse_line("5-9=-4") == ("5-9", -4.0)

    def test_float_answer(self):
        assert _parse_line("5+6=11.0") == ("5+6", 11.0)

    def test_missing_equals(self):
        assert _parse_line("5+6") == (None, None)

    def test_garbage(self):
        assert _parse_line("hello world") == (None, None)

    def test_empty(self):
        assert _parse_line("") == (None, None)


class TestParseNumberedLine:
    def test_basic(self):
        assert _parse_numbered_line("[1]5+6=11") == (1, "5+6", 11.0)

    def test_multidigit_index(self):
        assert _parse_numbered_line("[42]5-3=2") == (42, "5-3", 2.0)

    def test_whitespace_after_bracket_is_tolerated(self):
        assert _parse_numbered_line("[3] 5 + 6 = 11") == (3, "5+6", 11.0)

    def test_missing_bracket(self):
        assert _parse_numbered_line("1]5+6=11") == (None, None, None)

    def test_missing_answer(self):
        assert _parse_numbered_line("[1]5+6") == (None, None, None)

    def test_empty(self):
        assert _parse_numbered_line("") == (None, None, None)


class TestEvalExpr:
    def test_chain(self):
        assert _eval_expr("5+6-3") == 8

    def test_single_operand(self):
        assert _eval_expr("5") == 5

    def test_negative_result(self):
        assert _eval_expr("0-9") == -9

    def test_long_chain(self):
        assert _eval_expr("5+6+7+8-1-1") == 24


# 8-char hex UUID constants used throughout uuid tests.
# Lex order: "1..." < "3..." < "a..." so UUID_A < UUID_C < UUID_B.
UUID_A = "11111111"
UUID_B = "aaaaaaaa"
UUID_C = "33333333"


class TestParseNumberedUuidLine:
    def test_canonical_format(self):
        # Canonical format: "[N]hex,hex,...".
        idx, uuids = _parse_numbered_uuid_line(f"[7]{UUID_A},{UUID_B}")
        assert idx == 7
        assert uuids == [UUID_A, UUID_B]

    def test_parens_tolerated(self):
        # Parens around tokens still parse — \b boundaries pick them out.
        idx, uuids = _parse_numbered_uuid_line(f"[1]({UUID_A}),({UUID_B})")
        assert idx == 1
        assert uuids == [UUID_A, UUID_B]

    def test_uppercase_lowercased(self):
        idx, uuids = _parse_numbered_uuid_line(f"[3]{UUID_B.upper()}")
        assert idx == 3
        assert uuids == [UUID_B]

    def test_no_uuids_after_index(self):
        idx, uuids = _parse_numbered_uuid_line("[5]xyz")  # no 8-hex tokens
        assert idx == 5
        assert uuids == []

    def test_partial_hex_not_matched(self):
        # Only 7 hex chars — must not match.
        idx, uuids = _parse_numbered_uuid_line("[1]abcdef0,12345678")
        assert idx == 1
        assert uuids == ["12345678"]  # only the 8-char one

    def test_no_bracket_prefix(self):
        assert _parse_numbered_uuid_line("just some text") == (None, None)


# ─────────────────────────────────────────────────────────────────────────────
# score_response (unnumbered_streaming_sum)
# ─────────────────────────────────────────────────────────────────────────────

SUM_EXPRS = [
    {"expr": "5+6", "answer": 11},
    {"expr": "2-3", "answer": -1},
    {"expr": "1+2+3", "answer": 6},
]


class TestScoreResponse:
    def test_all_correct(self):
        out = "5+6=11\n2-3=-1\n1+2+3=6"
        assert score_response(out, SUM_EXPRS) == [(True, True, True)] * 3

    def test_missing_last_line(self):
        out = "5+6=11\n2-3=-1"
        scores = score_response(out, SUM_EXPRS)
        assert scores == [
            (True, True, True),
            (True, True, True),
            (False, False, False),
        ]

    def test_unparseable_lines_are_skipped(self):
        # Garbage lines drop out; remaining parseable lines align with expected.
        out = "5+6=11\nhello world\n2-3=-1\n!!\n1+2+3=6"
        assert score_response(out, SUM_EXPRS) == [(True, True, True)] * 3

    def test_wrong_expr_but_self_consistent(self):
        # Model wrote a different expression but did its OWN arithmetic right.
        out = "5+6=11\n2+3=5\n1+2+3=6"
        scores = score_response(out, SUM_EXPRS)
        assert scores[0] == (True, True, True)
        # Line 2: copy=False (2+3 != 2-3), answer=False (5 != -1),
        #         self_consistent=True (2+3 = 5 is correct arithmetic).
        assert scores[1] == (False, False, True)
        assert scores[2] == (True, True, True)

    def test_wrong_arithmetic_breaks_self_consistent(self):
        out = "5+6=99\n2-3=-1\n1+2+3=6"
        scores = score_response(out, SUM_EXPRS)
        # Line 0: copy=True, answer=False, self_consistent=False (5+6 != 99).
        assert scores[0] == (True, False, False)

    def test_empty_output(self):
        assert score_response("", SUM_EXPRS) == [(False, False, False)] * 3


# ─────────────────────────────────────────────────────────────────────────────
# score_response_numbered (streaming_sum + shuffled_streaming_sum)
# ─────────────────────────────────────────────────────────────────────────────


class TestScoreResponseNumbered:
    def test_in_order(self):
        out = "[1]5+6=11\n[2]2-3=-1\n[3]1+2+3=6"
        assert score_response_numbered(out, SUM_EXPRS) == [(True, True, True)] * 3

    def test_out_of_order_matches_by_number(self):
        out = "[3]1+2+3=6\n[1]5+6=11\n[2]2-3=-1"
        assert score_response_numbered(out, SUM_EXPRS) == [(True, True, True)] * 3

    def test_missing_number_marks_only_that_index(self):
        out = "[1]5+6=11\n[3]1+2+3=6"  # [2] absent
        scores = score_response_numbered(out, SUM_EXPRS)
        assert scores[0] == (True, True, True)
        assert scores[1] == (False, False, False)
        assert scores[2] == (True, True, True)

    def test_duplicate_number_first_wins(self):
        out = "[1]5+6=11\n[1]9+9=999\n[2]2-3=-1\n[3]1+2+3=6"
        scores = score_response_numbered(out, SUM_EXPRS)
        assert scores[0] == (True, True, True)  # first [1] won

    def test_unparseable_lines_dont_shift_alignment(self):
        out = "[1]5+6=11\ngarbage\n[2]2-3=-1\n[3]1+2+3=6"
        assert score_response_numbered(out, SUM_EXPRS) == [(True, True, True)] * 3

    def test_empty_output(self):
        assert score_response_numbered("", SUM_EXPRS) == [(False, False, False)] * 3


# ─────────────────────────────────────────────────────────────────────────────
# score_uuid_sort (streaming_uuid_sort + shuffled_streaming_uuid_sort)
# ─────────────────────────────────────────────────────────────────────────────

# uuid_lines = the canonical input-order UUIDs per line (NOT pre-sorted).
UUID_LINES = [
    [UUID_B, UUID_A, UUID_C],  # line 1 input
    [UUID_C, UUID_A],  # line 2 input
]
EXPECTED_LINE_1 = sorted(UUID_LINES[0])  # [A, C, B]
EXPECTED_LINE_2 = sorted(UUID_LINES[1])  # [A, C]


def _render_uuid_line(idx: int, uuids: list[str]) -> str:
    """Canonical render: [N]hex,hex,..."""
    return f"[{idx}]" + ",".join(uuids)


class TestScoreUuidSort:
    """score_uuid_sort returns per-line (copy_correct, answer_correct, self_consistent).

    copy_correct    : set of model's emitted UUIDs for this line equals
                      the set of UUIDs in the input for this line.
    answer_correct  : strict positional+length match against the expected
                      sorted list.
    self_consistent : model's emitted UUIDs are in non-decreasing lex order.
    """

    def test_all_correct(self):
        out = "\n".join(
            [
                _render_uuid_line(1, EXPECTED_LINE_1),
                _render_uuid_line(2, EXPECTED_LINE_2),
            ]
        )
        assert score_uuid_sort(out, UUID_LINES) == [
            (True, True, True),
            (True, True, True),
        ]

    def test_input_order_right_uuids_wrong_sort(self):
        # Model echoed input order (didn't sort). Set matches input,
        # answer_correct=False (wrong order), self_consistent=False (not sorted).
        out = "\n".join(
            [
                _render_uuid_line(1, UUID_LINES[0]),  # [B, A, C] — set OK, not sorted
                _render_uuid_line(2, EXPECTED_LINE_2),
            ]
        )
        scores = score_uuid_sort(out, UUID_LINES)
        assert scores[0] == (True, False, False)
        assert scores[1] == (True, True, True)

    def test_wrong_uuids_but_sorted_self_consistent_only(self):
        # Model emitted entirely different UUIDs but in lex order.
        # copy_correct=False (wrong set), answer_correct=False,
        # self_consistent=True (the wrong list is itself sorted).
        wrong_sorted = sorted(["bbbbbbbb", "cccccccc", "dddddddd"])
        out = "\n".join(
            [
                _render_uuid_line(1, wrong_sorted),
                _render_uuid_line(2, EXPECTED_LINE_2),
            ]
        )
        scores = score_uuid_sort(out, UUID_LINES)
        assert scores[0] == (False, False, True)
        assert scores[1] == (True, True, True)

    def test_missing_line_all_false(self):
        out = _render_uuid_line(1, EXPECTED_LINE_1)
        scores = score_uuid_sort(out, UUID_LINES)
        assert scores[0] == (True, True, True)
        assert scores[1] == (False, False, False)

    def test_short_line_breaks_copy_and_answer(self):
        # Model emitted only 2 of 3 expected UUIDs. Set differs (missing UUID_B),
        # so copy_correct=False. Length mismatch -> answer_correct=False.
        # The two emitted are in lex order so self_consistent=True.
        out = "\n".join(
            [
                _render_uuid_line(1, EXPECTED_LINE_1[:2]),
                _render_uuid_line(2, EXPECTED_LINE_2),
            ]
        )
        scores = score_uuid_sort(out, UUID_LINES)
        assert scores[0] == (False, False, True)
        assert scores[1] == (True, True, True)

    def test_extra_uuids_break_copy_and_answer(self):
        # Model added an UUID not in input. Set differs -> copy_correct=False.
        # Length differs -> answer_correct=False. Appended "ff..." keeps sort.
        out = "\n".join(
            [
                _render_uuid_line(1, EXPECTED_LINE_1 + ["ffffffff"]),
                _render_uuid_line(2, EXPECTED_LINE_2),
            ]
        )
        scores = score_uuid_sort(out, UUID_LINES)
        assert scores[0] == (False, False, True)
        assert scores[1] == (True, True, True)

    def test_answer_correct_implies_copy_and_self_consistent(self):
        # When answer_correct is True, copy_correct and self_consistent must
        # also be True by construction (expected = sorted(input)).
        out = "\n".join(
            [
                _render_uuid_line(1, EXPECTED_LINE_1),
                _render_uuid_line(2, EXPECTED_LINE_2),
            ]
        )
        for triple in score_uuid_sort(out, UUID_LINES):
            copy, ans, sc = triple
            if ans:
                assert copy and sc

    def test_uppercase_uuids_match(self):
        out = "\n".join(
            [
                _render_uuid_line(1, [u.upper() for u in EXPECTED_LINE_1]),
                _render_uuid_line(2, EXPECTED_LINE_2),
            ]
        )
        assert score_uuid_sort(out, UUID_LINES) == [
            (True, True, True),
            (True, True, True),
        ]

    def test_out_of_order_lines_matched_by_index(self):
        out = "\n".join(
            [
                _render_uuid_line(2, EXPECTED_LINE_2),
                _render_uuid_line(1, EXPECTED_LINE_1),
            ]
        )
        assert score_uuid_sort(out, UUID_LINES) == [
            (True, True, True),
            (True, True, True),
        ]

    def test_parens_tolerated_in_model_output(self):
        line_1 = "[1]" + ",".join(f"({u})" for u in EXPECTED_LINE_1)
        out = "\n".join([line_1, _render_uuid_line(2, EXPECTED_LINE_2)])
        assert score_uuid_sort(out, UUID_LINES) == [
            (True, True, True),
            (True, True, True),
        ]

    def test_empty_output(self):
        assert score_uuid_sort("", UUID_LINES) == [
            (False, False, False),
            (False, False, False),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# score_unnumbered_uuid_sort
# ─────────────────────────────────────────────────────────────────────────────


class TestScoreUnnumberedUuidSort:
    """score_unnumbered_uuid_sort: positional line matching, same per-line signals."""

    def test_all_correct(self):
        out = "\n".join(
            [
                ",".join(EXPECTED_LINE_1),
                ",".join(EXPECTED_LINE_2),
            ]
        )
        assert score_unnumbered_uuid_sort(out, UUID_LINES) == [
            (True, True, True),
            (True, True, True),
        ]

    def test_input_order_right_uuids_wrong_sort(self):
        out = "\n".join(
            [
                ",".join(UUID_LINES[0]),  # [B, A, C] — set OK, not sorted
                ",".join(EXPECTED_LINE_2),
            ]
        )
        scores = score_unnumbered_uuid_sort(out, UUID_LINES)
        assert scores[0] == (True, False, False)
        assert scores[1] == (True, True, True)

    def test_missing_line_all_false(self):
        out = ",".join(EXPECTED_LINE_1)
        scores = score_unnumbered_uuid_sort(out, UUID_LINES)
        assert scores[0] == (True, True, True)
        assert scores[1] == (False, False, False)

    def test_garbage_lines_skipped(self):
        # Non-UUID lines are skipped; remaining lines align positionally.
        out = "\n".join(
            [
                "some preamble text",
                ",".join(EXPECTED_LINE_1),
                ",".join(EXPECTED_LINE_2),
            ]
        )
        assert score_unnumbered_uuid_sort(out, UUID_LINES) == [
            (True, True, True),
            (True, True, True),
        ]

    def test_empty_output(self):
        assert score_unnumbered_uuid_sort("", UUID_LINES) == [
            (False, False, False),
            (False, False, False),
        ]

    def test_numbered_prefix_still_parsed(self):
        # If model outputs [N] prefix anyway, the 8-char hex tokens are still extracted.
        out = "\n".join(
            [
                f"[1]{','.join(EXPECTED_LINE_1)}",
                f"[2]{','.join(EXPECTED_LINE_2)}",
            ]
        )
        assert score_unnumbered_uuid_sort(out, UUID_LINES) == [
            (True, True, True),
            (True, True, True),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# _parse_csv_grid and score_csv_permutation
# ─────────────────────────────────────────────────────────────────────────────

# 2×3 grid with distinct cells.
CSV_GRID = [
    ["aa", "bb", "cc"],
    ["dd", "ee", "ff"],
]
# Identity permutation expected output.
CSV_EXPECTED_IDENTITY = ",[C0],[C1],[C2]\n[R0],aa,bb,cc\n[R1],dd,ee,ff"
# Row-swap (R1 first) + col-swap (C2, C0, C1).
CSV_EXPECTED_PERMUTED = ",[C2],[C0],[C1]\n[R1],ff,dd,ee\n[R0],cc,aa,bb"


class TestParseCsvGrid:
    def test_identity(self):
        grid = _parse_csv_grid(CSV_EXPECTED_IDENTITY, 2, 3)
        assert grid == [["aa", "bb", "cc"], ["dd", "ee", "ff"]]

    def test_permuted(self):
        grid = _parse_csv_grid(CSV_EXPECTED_PERMUTED, 2, 3)
        assert grid == [["ff", "dd", "ee"], ["cc", "aa", "bb"]]

    def test_no_header_row(self):
        # If the first line doesn't start with ',', it's treated as a data row.
        text = "[R0],aa,bb,cc\n[R1],dd,ee,ff"
        grid = _parse_csv_grid(text, 2, 3)
        assert grid == [["aa", "bb", "cc"], ["dd", "ee", "ff"]]

    def test_too_few_rows_padded(self):
        grid = _parse_csv_grid(CSV_EXPECTED_IDENTITY, 3, 3)
        assert len(grid) == 3
        assert grid[2] == ["", "", ""]

    def test_too_few_cols_padded(self):
        grid = _parse_csv_grid(CSV_EXPECTED_IDENTITY, 2, 4)
        assert grid[0] == ["aa", "bb", "cc", ""]

    def test_empty_input(self):
        grid = _parse_csv_grid("", 2, 3)
        assert grid == [["", "", ""], ["", "", ""]]


class TestScoreCsvPermutation:
    def test_perfect_match(self):
        cell_scores, row_scores, col_scores = score_csv_permutation(CSV_EXPECTED_PERMUTED, CSV_EXPECTED_PERMUTED, 2, 3)
        assert all(s[1] for s in cell_scores)
        assert all(row_scores)
        assert all(col_scores)

    def test_total_mismatch(self):
        wrong = ",[C0],[C1],[C2]\n[R0],xx,yy,zz\n[R1],pp,qq,rr"
        cell_scores, row_scores, col_scores = score_csv_permutation(wrong, CSV_EXPECTED_PERMUTED, 2, 3)
        assert not any(s[1] for s in cell_scores)
        assert not any(row_scores)
        assert not any(col_scores)

    def test_partial_row_correct(self):
        # First row correct, second row wrong.
        partial = (
            ",[C2],[C0],[C1]\n"
            "[R1],ff,dd,ee\n"
            "[R0],XX,aa,bb"  # first cell wrong
        )
        cell_scores, row_scores, col_scores = score_csv_permutation(partial, CSV_EXPECTED_PERMUTED, 2, 3)
        assert row_scores[0] is True
        assert row_scores[1] is False
        # col 0: cells (ff, XX) — second wrong → col False
        assert col_scores[0] is False
        # col 1: cells (dd, aa) — both correct → col True
        assert col_scores[1] is True

    def test_reward_is_cell_accuracy(self):
        # 5 of 6 cells correct → reward = 5/6.
        partial = ",[C2],[C0],[C1]\n[R1],ff,dd,ee\n[R0],XX,aa,bb"
        cell_scores, _, _ = score_csv_permutation(partial, CSV_EXPECTED_PERMUTED, 2, 3)
        flat = [s[1] for s in cell_scores]
        assert sum(flat) == 5
        assert len(flat) == 6

    def test_empty_model_output(self):
        cell_scores, row_scores, col_scores = score_csv_permutation("", CSV_EXPECTED_PERMUTED, 2, 3)
        assert not any(s[1] for s in cell_scores)
        assert not any(row_scores)
        assert not any(col_scores)

    def test_n_items_equals_n_rows_times_n_cols(self):
        cell_scores, row_scores, col_scores = score_csv_permutation(CSV_EXPECTED_PERMUTED, CSV_EXPECTED_PERMUTED, 2, 3)
        assert len(cell_scores) == 2 * 3
        assert len(row_scores) == 2
        assert len(col_scores) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Variable expansion (var_expand types)
# ─────────────────────────────────────────────────────────────────────────────

# Canonical (1-indexed by position) expected items for var-expand tests.
VAR_EXPRS = [
    {"expr": "a3f+7b2", "answer": "big dog"},
    {"expr": "0c1+9e4", "answer": "black cat"},
    {"expr": "1a2+3b4", "answer": "red fox"},
]


class TestNormalizeValue:
    def test_collapses_whitespace_and_lowercases(self):
        assert _normalize_value("Big   Dog") == "big dog"
        assert _normalize_value("  black\tcat ") == "black cat"

    def test_already_normalized(self):
        assert _normalize_value("red fox") == "red fox"


class TestParseNumberedValueLine:
    def test_basic(self):
        assert _parse_numbered_value_line("[1]big dog") == (1, "big dog")

    def test_multidigit_index_and_whitespace(self):
        assert _parse_numbered_value_line("[42] black cat ") == (42, "black cat")

    def test_no_bracket(self):
        assert _parse_numbered_value_line("big dog") == (None, None)

    def test_empty_value(self):
        assert _parse_numbered_value_line("[3]") == (3, "")


class TestScoreVarExpandNumbered:
    def test_all_correct(self):
        out = "[1]big dog\n[2]black cat\n[3]red fox"
        assert score_var_expand_numbered(out, VAR_EXPRS) == [(True, True, True)] * 3

    def test_out_of_order_matches_by_number(self):
        out = "[3]red fox\n[1]big dog\n[2]black cat"
        assert score_var_expand_numbered(out, VAR_EXPRS) == [(True, True, True)] * 3

    def test_missing_number_marks_only_that_index(self):
        out = "[1]big dog\n[3]red fox"  # [2] absent
        scores = score_var_expand_numbered(out, VAR_EXPRS)
        assert scores == [
            (True, True, True),
            (False, False, False),
            (True, True, True),
        ]

    def test_wrong_value_scores_false(self):
        out = "[1]big dog\n[2]white cat\n[3]red fox"
        scores = score_var_expand_numbered(out, VAR_EXPRS)
        assert scores[1] == (False, False, False)

    def test_whitespace_and_case_tolerated(self):
        out = "[1]Big   Dog\n[2]BLACK cat\n[3]red  fox"
        assert score_var_expand_numbered(out, VAR_EXPRS) == [(True, True, True)] * 3

    def test_duplicate_number_first_wins(self):
        out = "[1]big dog\n[1]wrong words\n[2]black cat\n[3]red fox"
        scores = score_var_expand_numbered(out, VAR_EXPRS)
        assert scores[0] == (True, True, True)

    def test_empty_output(self):
        assert score_var_expand_numbered("", VAR_EXPRS) == [(False, False, False)] * 3


class TestScoreVarExpandUnnumbered:
    def test_all_correct(self):
        out = "big dog\nblack cat\nred fox"
        assert score_var_expand_unnumbered(out, VAR_EXPRS) == [(True, True, True)] * 3

    def test_positional_matching(self):
        out = "big dog\nred fox\nblack cat"  # order 2 and 3 swapped -> both wrong
        scores = score_var_expand_unnumbered(out, VAR_EXPRS)
        assert scores[0] == (True, True, True)
        assert scores[1] == (False, False, False)
        assert scores[2] == (False, False, False)

    def test_missing_last_line(self):
        out = "big dog\nblack cat"
        scores = score_var_expand_unnumbered(out, VAR_EXPRS)
        assert scores == [
            (True, True, True),
            (True, True, True),
            (False, False, False),
        ]

    def test_blank_lines_skipped(self):
        out = "big dog\n\nblack cat\n\nred fox"
        assert score_var_expand_unnumbered(out, VAR_EXPRS) == [(True, True, True)] * 3

    def test_empty_output(self):
        assert score_var_expand_unnumbered("", VAR_EXPRS) == [(False, False, False)] * 3


@pytest.mark.parametrize("output", ["1+2=..", "1+2=1.2.3"])
def test_invalid_numeric_answer(output: str) -> None:
    assert _parse_line(output) == (None, None)
    assert _parse_numbered_line(f"[1]{output}") == (None, None, None)


def test_invalid_expression() -> None:
    assert _eval_expr("1++2") is None


def test_oversized_line_numbers_are_unparseable() -> None:
    # Python limits decimal-to-int conversion; malformed model output must not crash scoring.
    index = "9" * (sys.get_int_max_str_digits() + 5000)
    assert _parse_numbered_line(f"[{index}]1+2=3") == (None, None, None)
    assert _parse_numbered_uuid_line(f"[{index}]aaaaaaaa") == (None, None)
    assert _parse_numbered_value_line(f"[{index}]big dog") == (None, None)


@pytest.mark.parametrize("use_dataset", [False, True])
def test_arithmetic_scoring_cli(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str], use_dataset: bool
) -> None:
    sample = {"expressions": [{"expr": "1+2", "answer": 3}]}
    sample_path = tmp_path / "sample.jsonl"
    sample_path.write_text(json.dumps(sample) + "\n")
    output_path = tmp_path / "output.txt"
    output_path.write_text("1+2=3")
    flag = "--dataset" if use_dataset else "--sample"
    monkeypatch.setattr(sys, "argv", ["parse.py", "--output", str(output_path), flag, str(sample_path)])
    parse.main()
    assert "Answer correct:      1/1" in capsys.readouterr().out


def test_cli_requires_ground_truth(tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    output = tmp_path / "output.txt"
    output.write_text("1+2=3")
    monkeypatch.setattr(sys, "argv", ["parse.py", "--output", str(output)])
    with pytest.raises(SystemExit, match="1"):
        parse.main()
    assert "provide --sample or --dataset" in capsys.readouterr().err


def test_dataset_sample_index_out_of_range(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    with pytest.raises(IndexError, match="Sample index 1 not found"):
        parse._load_sample(dataset, 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
