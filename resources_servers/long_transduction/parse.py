#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score model responses against long_transduction dataset ground truth.

Supports five sample types — see `_SCORERS_BY_TYPE` in app.py for dispatch:
  - unnumbered_streaming_sum    : score_response
  - streaming_sum               : score_response_numbered
  - shuffled_streaming_sum      : score_response_numbered
  - streaming_uuid_sort         : score_uuid_sort
  - shuffled_streaming_uuid_sort: score_uuid_sort
  - unnumbered_var_expand       : score_var_expand_unnumbered
  - streaming_var_expand        : score_var_expand_numbered
  - shuffled_streaming_var_expand: score_var_expand_numbered

Usage (as a library):
    from parse import score_response, score_uuid_sort
    arithmetic_scores = score_response(model_output, sample["expressions"])
    uuid_scores       = score_uuid_sort(model_output, sample["uuid_lines"])

Usage (CLI):
    python parse.py --output model_output.txt --sample sample.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional


def _normalize_expr(expr: str) -> str:
    """Strip all whitespace: '5 + 6 - 3' and '5+6-3' both become '5+6-3'."""
    return re.sub(r"\s+", "", expr.strip())


def _parse_line(line: str) -> tuple[Optional[str], Optional[float]]:
    """Parse one output line of the form '<expr>=<number>'.

    Tolerates spaces around operators and =, and float answers like 11.0.
    Handles subtraction and negative results.
    Returns (normalized_expr, answer) or (None, None) if unparseable.
    """
    line = line.strip()
    match = re.match(r"^([\d\s+\-]+)=\s*(-?[\d.]+)\s*$", line)
    if not match:
        return None, None
    try:
        answer = float(match.group(2))
    except ValueError:
        return None, None
    return _normalize_expr(match.group(1)), answer


_NUMBERED_LINE_RE = re.compile(r"^\[(\d+)\]\s*([\d\s+\-]+)=\s*(-?[\d.]+)\s*$")


def _parse_numbered_line(
    line: str,
) -> tuple[Optional[int], Optional[str], Optional[float]]:
    """Parse one output line of the form '[N]<expr>=<number>'.

    Returns (index, normalized_expr, answer) or (None, None, None) if the
    line does not match. Whitespace after the closing bracket is tolerated.
    """
    line = line.strip()
    match = _NUMBERED_LINE_RE.match(line)
    if not match:
        return None, None, None
    try:
        idx = int(match.group(1))
        answer = float(match.group(3))
    except ValueError:
        return None, None, None
    return idx, _normalize_expr(match.group(2)), answer


def _eval_expr(expr: str) -> Optional[int]:
    """Evaluate a normalized expression like '5+6-3' -> 8."""
    try:
        tokens = re.split(r"(?=[+\-])", expr)
        return sum(int(t) for t in tokens if t)
    except ValueError:
        return None


def score_response(
    model_output: str,
    expressions: list[dict],
) -> list[tuple[bool, bool, bool]]:
    """Score a raw model response against the expected expressions (legacy, unnumbered).

    Args:
        model_output: raw text output from the model.
        expressions: list of {"expr": str, "answer": int} from the dataset sample.

    Returns:
        List of (expr_correct, answer_correct, self_consistent) tuples, one per
        expected expression. All False when the line is missing or unparseable.

        expr_correct:    model copied the expression exactly (whitespace-normalized)
        answer_correct:  model's answer matches the correct answer to the original expression
        self_consistent: model's answer matches the correct answer to the expression it wrote
    """
    parsed_lines = []
    for line in model_output.split("\n"):
        if line.strip():
            expr, answer = _parse_line(line)
            if answer is not None:
                parsed_lines.append((expr, answer))

    scores: list[tuple[bool, bool, bool]] = []

    for i, expected in enumerate(expressions):
        if i >= len(parsed_lines):
            scores.append((False, False, False))
            continue

        parsed_expr, parsed_answer = parsed_lines[i]

        expr_ok = _normalize_expr(expected["expr"]) == parsed_expr
        answer_ok = abs(parsed_answer - expected["answer"]) < 1e-9
        written_answer = _eval_expr(parsed_expr)
        self_consistent = written_answer is not None and abs(parsed_answer - written_answer) < 1e-9

        scores.append((expr_ok, answer_ok, self_consistent))

    return scores


def score_response_numbered(
    model_output: str,
    expressions: list[dict],
) -> list[tuple[bool, bool, bool]]:
    """Score model output for the numbered '[N]expr=answer' format.

    The `expressions` list is canonical (1-indexed by list-position+1). This
    function looks up each expected expression by its number in the model
    output, so it works identically for "streaming_sum" (input in numerical
    order) and "shuffled_streaming_sum" (input in shuffled order) — the model
    is expected to emit output in ascending numerical order in either case,
    but matching is by [N] so a partially-mis-ordered output still scores its
    correctly-numbered lines.

    Returns one (expr_correct, answer_correct, self_consistent) per expected
    expression; all-False when no line with that number was found.
    """
    by_number: dict[int, tuple[str, float]] = {}
    for line in model_output.split("\n"):
        if not line.strip():
            continue
        idx, expr, answer = _parse_numbered_line(line)
        if idx is None or answer is None:
            continue
        # If a number is duplicated in the output, keep the first occurrence —
        # that's what the model intended at that point in its stream.
        if idx not in by_number:
            by_number[idx] = (expr, answer)

    scores: list[tuple[bool, bool, bool]] = []
    for i, expected in enumerate(expressions):
        number = i + 1  # 1-indexed
        entry = by_number.get(number)
        if entry is None:
            scores.append((False, False, False))
            continue
        parsed_expr, parsed_answer = entry
        expr_ok = _normalize_expr(expected["expr"]) == parsed_expr
        answer_ok = abs(parsed_answer - expected["answer"]) < 1e-9
        written_answer = _eval_expr(parsed_expr)
        self_consistent = written_answer is not None and abs(parsed_answer - written_answer) < 1e-9
        scores.append((expr_ok, answer_ok, self_consistent))
    return scores


# 8-char hex token (the first segment of a uuid4). \b boundaries prevent
# matching mid-word, so adjacent commas or end-of-string anchor each token.
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}\b")

_NUMBERED_PREFIX_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")


def _parse_numbered_uuid_line(
    line: str,
) -> tuple[Optional[int], Optional[List[str]]]:
    """Parse a line of form '[N]hex,hex,...' (8-char hex tokens).

    Returns (index, [hex_lowercase, ...]) — uses _UUID_RE.findall so optional
    whitespace, parens, brackets, or quotes around tokens are tolerated.
    Returns (None, None) if the line has no [N] prefix.
    """
    s = line.strip()
    m = _NUMBERED_PREFIX_RE.match(s)
    if not m:
        return None, None
    try:
        idx = int(m.group(1))
    except ValueError:
        return None, None
    rest = m.group(2)
    uuids = [u.lower() for u in _UUID_RE.findall(rest)]
    return idx, uuids


def score_uuid_sort(
    model_output: str,
    uuid_lines: list[list[str]],
) -> list[tuple[bool, bool, bool]]:
    """Score the per-line UUID-sort task.

    Args:
        model_output: raw model text.
        uuid_lines: list of lines, each a list of hex tokens in their CANONICAL
            (input-presentation) order. The expected per-line output is the
            same tokens sorted lexicographically. Position in this list defines
            the [N] index (line 0 -> "[1]", etc.).

    Returns:
        One (copy_correct, answer_correct, self_consistent) tuple per expected
        line — same per-position shape as `score_response*` for arithmetic:

          copy_correct    : the set of UUIDs the model emitted for this line
                            equals the set of UUIDs in the input for this line
                            (no missing, no extras). Order and duplicates do
                            not affect this signal.
          answer_correct  : the model's emitted list exactly matches the
                            expected sorted list — same length, same UUIDs,
                            in the right order.
          self_consistent : the UUIDs the model wrote at this line are
                            themselves in non-decreasing lex order, even if
                            those UUIDs are not the expected ones. Isolates
                            "did the model sort what it has?" from "did the
                            model copy the right UUIDs?".

        A missing line (no matching [N] in model output) yields
        (False, False, False). answer_correct=True implies copy_correct=True
        and self_consistent=True (the expected list is itself sorted).
    """
    expected_per_line = [sorted(u.lower() for u in uuids) for uuids in uuid_lines]
    input_sets = [set(u.lower() for u in uuids) for uuids in uuid_lines]

    by_number: dict[int, list[str]] = {}
    for line in model_output.split("\n"):
        if not line.strip():
            continue
        idx, uuids = _parse_numbered_uuid_line(line)
        if idx is None or uuids is None or not uuids:
            continue
        if idx not in by_number:  # first occurrence wins on duplicates
            by_number[idx] = uuids

    scores: list[tuple[bool, bool, bool]] = []
    for i, expected_sorted in enumerate(expected_per_line):
        model_uuids = by_number.get(i + 1)
        if not model_uuids:
            scores.append((False, False, False))
            continue
        copy_correct = set(model_uuids) == input_sets[i]
        answer_correct = len(model_uuids) == len(expected_sorted) and all(
            model_uuids[j] == expected_sorted[j] for j in range(len(expected_sorted))
        )
        self_consistent = all(model_uuids[j] <= model_uuids[j + 1] for j in range(len(model_uuids) - 1))
        scores.append((copy_correct, answer_correct, self_consistent))
    return scores


def score_unnumbered_uuid_sort(
    model_output: str,
    uuid_lines: list[list[str]],
) -> list[tuple[bool, bool, bool]]:
    """Score the unnumbered per-line UUID-sort task (positional matching).

    Lines in the model output are matched positionally to expected lines.
    Non-empty lines with no 8-char hex tokens are skipped so that stray
    punctuation or prose doesn't shift alignment.

    Returns one (copy_correct, answer_correct, self_consistent) per expected
    line — same semantics as score_uuid_sort.
    """
    expected_per_line = [sorted(u.lower() for u in uuids) for uuids in uuid_lines]
    input_sets = [set(u.lower() for u in uuids) for uuids in uuid_lines]

    parsed_lines: list[list[str]] = []
    for line in model_output.split("\n"):
        if not line.strip():
            continue
        uuids = [u.lower() for u in _UUID_RE.findall(line)]
        if uuids:
            parsed_lines.append(uuids)

    scores: list[tuple[bool, bool, bool]] = []
    for i, expected_sorted in enumerate(expected_per_line):
        if i >= len(parsed_lines):
            scores.append((False, False, False))
            continue
        model_uuids = parsed_lines[i]
        copy_correct = set(model_uuids) == input_sets[i]
        answer_correct = len(model_uuids) == len(expected_sorted) and all(
            model_uuids[j] == expected_sorted[j] for j in range(len(expected_sorted))
        )
        self_consistent = all(model_uuids[j] <= model_uuids[j + 1] for j in range(len(model_uuids) - 1))
        scores.append((copy_correct, answer_correct, self_consistent))
    return scores


def _normalize_value(s: str) -> str:
    """Collapse internal whitespace to single spaces, strip edges, lowercase.

    'Big   Dog' and 'big dog' both become 'big dog'. Used to compare a
    resolved 'adjective noun' value against the expected string so trivial
    spacing/casing differences don't count as errors.
    """
    return " ".join(s.split()).lower()


_NUMBERED_VALUE_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")


def _parse_numbered_value_line(line: str) -> tuple[Optional[int], Optional[str]]:
    """Parse a line of the form '[N]<value>' into (index, value).

    Returns (None, None) if there is no '[N]' prefix. The value is returned
    verbatim (still needs _normalize_value before comparison).
    """
    m = _NUMBERED_VALUE_RE.match(line.strip())
    if not m:
        return None, None
    try:
        idx = int(m.group(1))
    except ValueError:
        return None, None
    return idx, m.group(2)


def score_var_expand_numbered(
    model_output: str,
    expressions: list[dict],
) -> list[tuple[bool, bool, bool]]:
    """Score the numbered variable-expansion task ('[N]<word> <word>').

    Each expected item is {"expr": "aaa+bbb", "answer": "big dog"}; the answer
    is the two resolved words separated by a space. The `expressions` list is
    canonical (1-indexed by position), so this works identically for
    "streaming_var_expand" (input in order) and "shuffled_streaming_var_expand"
    (input shuffled) — matching is by [N].

    Returns one (copy_correct, answer_correct, self_consistent) per expected
    item. Since the output carries no expression to echo, the three signals are
    identical — copy_correct collapses onto answer_correct — but the 3-bool
    shape is kept uniform with the other scorers. All-False for a missing [N].
    """
    by_number: dict[int, str] = {}
    for line in model_output.split("\n"):
        if not line.strip():
            continue
        idx, value = _parse_numbered_value_line(line)
        if idx is None:
            continue
        if idx not in by_number:  # first occurrence wins on duplicates
            by_number[idx] = value

    scores: list[tuple[bool, bool, bool]] = []
    for i, expected in enumerate(expressions):
        value = by_number.get(i + 1)
        if value is None:
            scores.append((False, False, False))
            continue
        ok = _normalize_value(value) == _normalize_value(str(expected["answer"]))
        scores.append((ok, ok, ok))
    return scores


def score_var_expand_unnumbered(
    model_output: str,
    expressions: list[dict],
) -> list[tuple[bool, bool, bool]]:
    """Score the unnumbered variable-expansion task by positional line matching.

    Each non-empty model-output line is matched positionally to the expected
    items. Returns one (copy_correct, answer_correct, self_consistent) per
    expected item — the three signals are identical (see
    score_var_expand_numbered). All-False for a missing line.
    """
    parsed = [line.strip() for line in model_output.split("\n") if line.strip()]
    scores: list[tuple[bool, bool, bool]] = []
    for i, expected in enumerate(expressions):
        if i >= len(parsed):
            scores.append((False, False, False))
            continue
        ok = _normalize_value(parsed[i]) == _normalize_value(str(expected["answer"]))
        scores.append((ok, ok, ok))
    return scores


def _parse_csv_grid(text: str, n_rows: int, n_cols: int) -> list[list[str]]:
    """Parse a CSV with optional row/col headers into an n_rows × n_cols cell grid.

    The first line is treated as a header and skipped when it starts with ','.
    Each data line's first comma-separated token is the row label and is skipped.
    Missing cells are padded with empty strings.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    start = 1 if (lines and lines[0].startswith(",")) else 0
    grid: list[list[str]] = []
    for line in lines[start : start + n_rows]:
        parts = line.split(",")
        cells = [c.strip() for c in parts[1 : n_cols + 1]]
        while len(cells) < n_cols:
            cells.append("")
        grid.append(cells)
    while len(grid) < n_rows:
        grid.append([""] * n_cols)
    return grid


def score_csv_permutation(
    model_output: str,
    expected_output: str,
    n_rows: int,
    n_cols: int,
) -> tuple[list[list[bool]], list[bool], list[bool]]:
    """Score the CSV permutation task against the expected permuted CSV.

    Args:
        model_output:    raw model text.
        expected_output: the correct permuted CSV string (stored in the dataset row).
        n_rows, n_cols:  grid dimensions.

    Returns:
        cell_item_scores: n_rows*n_cols items, each [cell_correct, cell_correct, cell_correct].
            Index 1 is used as the reward signal, so cell accuracy drives the reward.
        row_scores:  n_rows bools — True when every cell in that row matches.
        col_scores:  n_cols bools — True when every cell in that column matches.
    """
    model_grid = _parse_csv_grid(model_output, n_rows, n_cols)
    expected_grid = _parse_csv_grid(expected_output, n_rows, n_cols)

    correct = [[model_grid[i][j] == expected_grid[i][j] for j in range(n_cols)] for i in range(n_rows)]

    cell_item_scores = [[correct[i][j], correct[i][j], correct[i][j]] for i in range(n_rows) for j in range(n_cols)]
    row_scores = [all(correct[i][j] for j in range(n_cols)) for i in range(n_rows)]
    col_scores = [all(correct[i][j] for i in range(n_rows)) for j in range(n_cols)]
    return cell_item_scores, row_scores, col_scores


def _load_sample(dataset_path: Path, index: int) -> dict:
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Sample index {index} not found in {dataset_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a model response for long_sum.")
    parser.add_argument("--output", required=True, help="File containing raw model output text.")
    parser.add_argument("--dataset", help="Path to a long_sum JSONL dataset file.")
    parser.add_argument("--index", type=int, default=0, help="Which sample to score against (default: 0).")
    parser.add_argument("--sample", help="Path to a single JSON sample file (alternative to --dataset/--index).")
    args = parser.parse_args()

    model_output = Path(args.output).read_text()

    if args.sample:
        sample = json.loads(Path(args.sample).read_text())
    elif args.dataset:
        sample = _load_sample(Path(args.dataset), args.index)
    else:
        print("Error: provide --sample or --dataset.", file=sys.stderr)
        sys.exit(1)

    scores = score_response(model_output, sample["expressions"])

    total = len(scores)
    print(f"Expression correct:  {sum(e for e, _, __ in scores)}/{total}")
    print(f"Answer correct:      {sum(a for _, a, __ in scores)}/{total}")
    print(f"Self-consistent:     {sum(s for _, __, s in scores)}/{total}")
    print(f"Per-expression: {scores}")


if __name__ == "__main__":
    main()
