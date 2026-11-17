# SPDX-FileCopyrightText: Copyright (2025) Databricks, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from databricks/officeqa reward.py at revision:
# 7b9a3c154ef9fb40215bb67934afc43e6799de16
# https://github.com/databricks/officeqa/blob/7b9a3c154ef9fb40215bb67934afc43e6799de16/reward.py
# Includes materials developed by Databricks, Inc. (https://www.databricks.com/).
# Licensed under Apache-2.0; see the repository root LICENSE.
# Changes: NVIDIA header, formatting and type annotations; scoring behavior is preserved.

"""OfficeQA Pro scoring. The resources server strips inline reasoning before calling it."""

import re


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent parsing.

    - Convert Unicode minus (−, \u2212) to ASCII hyphen (-)
    - Remove extra whitespace

    Raises:
        ValueError: If text is None or empty
    """
    if not text:
        raise ValueError("Cannot normalize empty or None text")

    normalized = text.replace("\u2212", "-")
    return re.sub(r"\s+", " ", normalized).strip()


_CURRENCY_SYMBOLS = r"$£€¥₹¢₩₽"
_NUMBER_BODY = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"
_VALID_THOUSANDS_RE = r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)"
_LIST_NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def _normalize_numeric_formatting(text: str) -> str:
    def _accounting_repl(match: re.Match) -> str:
        currency = match.group(1)
        number = match.group(2)
        if currency is None and is_likely_year(float(number.replace(",", ""))):
            return match.group(0)
        return f"-{number}"

    text = re.sub(
        rf"\(\s*([{_CURRENCY_SYMBOLS}])?\s*({_NUMBER_BODY})\s*\)",
        _accounting_repl,
        text,
    )
    return re.sub(rf"[{_CURRENCY_SYMBOLS}]", "", text)


def extract_numbers_with_context(text: str) -> list[tuple[float, str, bool, bool]]:
    """
    Extract numbers with surrounding context for unit detection.

    Returns list of tuples: (number_value, context_string, has_percent, is_negative)

    Raises:
        ValueError: If text parsing fails unexpectedly
    """
    if not text:
        raise ValueError("Cannot extract numbers from empty text")

    # Normalize text first
    text = normalize_text(text)
    text = _normalize_numeric_formatting(text)

    # Remove commas only from thousands-separated numbers (e.g. 1,000,000 -> 1000000)
    text_no_commas = re.sub(
        _VALID_THOUSANDS_RE,
        lambda m: m.group().replace(",", ""),
        text,
    )

    numbers_with_context = []

    # Match negative numbers, decimals, and percentages
    # Pattern: optional minus, digits, optional decimal part, optional %
    pattern = r"-?\d+\.?\d*%?"

    for match in re.finditer(pattern, text_no_commas):
        matched_text = match.group()
        if not matched_text or matched_text == "-":
            continue

        # Check for percentage
        has_percent = matched_text.endswith("%")
        num_text = matched_text.rstrip("%")

        # Check for negative
        is_negative = num_text.startswith("-")

        try:
            num = float(num_text)
        except ValueError as e:
            raise ValueError(f"Failed to parse number from '{matched_text}': {e}") from e

        # Get context (20 chars before and after)
        start = max(0, match.start() - 20)
        end = min(len(text_no_commas), match.end() + 20)
        context = text_no_commas[start:end].lower()

        numbers_with_context.append((num, context, has_percent, is_negative))

    return numbers_with_context


def _single_bracketed_list_body(text: str) -> str | None:
    """Return the body of one bracketed list if surrounding text is label-like."""
    stripped = normalize_text(text)
    matches = list(re.finditer(r"\[([^\[\]]+)\]", stripped))
    if len(matches) != 1:
        return None

    match = matches[0]
    outside = (stripped[: match.start()] + stripped[match.end() :]).strip()
    if re.search(r"\d", outside):
        return None
    return match.group(1)


def _numeric_list_body(text: str) -> str:
    return _single_bracketed_list_body(text) or normalize_text(text)


def _parse_numeric_list_item(raw_item: str) -> float | None:
    item = raw_item.strip()
    if not item:
        return None
    if "," in item and re.search(r"\s,|,\s", item):
        return None

    item = _normalize_numeric_formatting(normalize_text(item))
    if _LIST_NUMBER_RE.fullmatch(item) is None:
        return None

    return float(item.rstrip("%").replace(",", ""))


def _comma_chunk_spans(body: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(",", body):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(body)))
    return spans


def _segment_numeric_list_values(text: str, expected_len: int) -> list[list[float]]:
    """
    Parse comma-delimited numeric-list candidates with exactly expected_len items.

    This intentionally runs only in the gold-list path. It lets a prediction
    like ``[8,152,260]`` match a two-item gold list ``[8, 152260]`` without
    changing global scalar parsing of the valid thousands number ``8,152,260``.
    """
    if expected_len < 2:
        return []

    body = _numeric_list_body(text)
    spans = _comma_chunk_spans(body)
    if len(spans) < expected_len:
        return []

    memo: dict[tuple[int, int], list[list[float]]] = {}

    def dfs(chunk_index: int, remaining: int) -> list[list[float]]:
        key = (chunk_index, remaining)
        if key in memo:
            return memo[key]

        if remaining == 0:
            return [[]] if chunk_index == len(spans) else []

        results: list[list[float]] = []
        max_end = len(spans) - remaining + 1
        for end_index in range(chunk_index + 1, max_end + 1):
            start = spans[chunk_index][0]
            end = spans[end_index - 1][1]
            value = _parse_numeric_list_item(body[start:end])
            if value is None:
                continue
            for tail in dfs(end_index, remaining - 1):
                results.append([value, *tail])

        memo[key] = results
        return results

    deduped: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    for values in dfs(0, expected_len):
        key = tuple(values)
        if key not in seen:
            deduped.append(values)
            seen.add(key)
    return deduped


def _ground_truth_bracketed_numeric_list(
    ground_truth: str,
    gt_numbers_with_context: list[tuple[float, str, bool, bool]],
) -> list[tuple[float, str]] | None:
    body = _single_bracketed_list_body(ground_truth)
    if body is None:
        return None

    gt_numbers = [(num, ctx) for num, ctx, _, _ in gt_numbers_with_context]
    if len(gt_numbers) < 2:
        return None

    candidate_values = _segment_numeric_list_values(body, len(gt_numbers))
    gt_values = [num for num, _ in gt_numbers]
    if not any(candidate == gt_values for candidate in candidate_values):
        return None

    return gt_numbers


def _ordered_numbers_match(
    gt_numbers: list[tuple[float, str]],
    pred_numbers: list[tuple[float, str]],
    tolerance: float,
) -> bool:
    if len(gt_numbers) != len(pred_numbers):
        return False

    for (gt_val, gt_context), (pred_val, pred_context) in zip(gt_numbers, pred_numbers):
        gt_base, gt_unit = normalize_number_with_units(gt_val, gt_context)
        pred_base, pred_unit = normalize_number_with_units(pred_val, pred_context)
        if not units_compatible(gt_unit, pred_unit):
            return False
        if gt_base == 0:
            if pred_base != 0:
                return False
            continue
        if abs(gt_base - pred_base) / abs(gt_base) > tolerance:
            return False

    return True


def _match_bracketed_numeric_list(
    ground_truth: str,
    predicted: str,
    gt_numbers_with_context: list[tuple[float, str, bool, bool]],
    pred_numbers_with_context: list[tuple[float, str, bool, bool]],
    tolerance: float,
) -> tuple[bool, str] | None:
    gt_numbers = _ground_truth_bracketed_numeric_list(ground_truth, gt_numbers_with_context)
    if gt_numbers is None:
        return None

    segmented_candidates = _segment_numeric_list_values(predicted, len(gt_numbers))
    for values in segmented_candidates:
        pred_candidate = [(value, "") for value in values]
        if _ordered_numbers_match(gt_numbers, pred_candidate, tolerance):
            return True, f"Ordered list match: segmented prediction into {len(values)} numbers"

    pred_numbers = [(num, ctx) for num, ctx, _, _ in pred_numbers_with_context]
    pred_numbers = _filter_context_years_for_direct_answer(gt_numbers, pred_numbers, ground_truth)
    if _ordered_numbers_match(gt_numbers, pred_numbers, tolerance):
        return True, f"Ordered list match: All {len(gt_numbers)} numbers matched in order"

    expected = [num for num, _ in gt_numbers]
    observed = [[num for num, _ in pred_numbers]]
    observed.extend(segmented_candidates)
    return False, f"Ordered list mismatch: expected {expected}, observed candidates {observed[:5]}"


def detect_unit_in_context(context: str) -> tuple[str | None, float]:
    """
    Detect unit words in context and return multiplier.

    Returns: (unit_name, multiplier)
        unit_name: 'trillion', 'billion', 'million', 'thousand', or None
        multiplier: corresponding multiplier or 1.0

    Supports plural forms: millions, billions, trillions, thousands
    """
    context_lower = context.lower()

    # Check for trillion/trillions
    if re.search(r"\btrillions?\b", context_lower):
        return ("trillion", 1e12)

    # Check for billion/billions
    if re.search(r"\bbillions?\b", context_lower) or re.search(r"\bb\b", context_lower):
        return ("billion", 1e9)

    # Check for million/millions
    if re.search(r"\bmillions?\b", context_lower) or re.search(r"\bm\b", context_lower):
        return ("million", 1e6)

    # Check for thousand/thousands
    if re.search(r"\bthousands?\b", context_lower) or re.search(r"\bk\b", context_lower):
        return ("thousand", 1e3)

    return (None, 1.0)


def normalize_number_with_units(number: float, context: str) -> tuple[float, str | None]:
    """
    Normalize a number based on its unit context.

    If the context contains "543 million", return (543, 'million')
    This represents the BASE number before unit application.

    Returns: (base_number, unit_name)

    Raises:
        ValueError: If normalization fails
    """
    try:
        unit_name, _ = detect_unit_in_context(context)

        # The number in text is already the BASE number
        # e.g., "543 million" means base=543, unit=million
        # We do NOT multiply here - that's for comparison later

        return (number, unit_name)
    except Exception as e:
        raise ValueError(f"Failed to normalize number {number} with context '{context}': {e}") from e


def units_compatible(gt_unit: str | None, pred_unit: str | None) -> bool:
    return gt_unit is None or pred_unit is None or gt_unit == pred_unit


def is_likely_year(num: float) -> bool:
    """
    Check if a number is likely a year (1900-2100 range).

    Note: This is used to filter out incidental year references in predictions
    (e.g., "reported in 2023") when the ground truth is clearly not a year.
    However, we DO NOT filter when:
    - Ground truth itself is a year-like number (could be a dollar value like "2003")
    - Ground truth contains text (e.g., "March 1977" - the year should match)
    """
    return 1900 <= num <= 2100 and num == int(num)


def has_significant_text(text: str) -> tuple[bool, str]:
    """
    Check if text has significant non-numeric content beyond unit words.

    Returns: (has_significant_text, cleaned_text)
        has_significant_text: True if there's meaningful text beyond numbers/units
        cleaned_text: The text with numbers and common unit words removed

    Examples:
        "March 1977" → (True, "march")
        "543 million" → (False, "")
        "April 15, 2020" → (True, "april")
        "1234" → (False, "")
    """
    if not text:
        return False, ""

    # Normalize and lowercase
    cleaned = normalize_text(text).lower()

    # Remove numbers (including decimals, percentages, commas)
    cleaned = re.sub(r"-?\d+\.?\d*%?", "", cleaned)
    cleaned = re.sub(r"[,]", "", cleaned)

    # Remove common unit words
    unit_words = [
        "trillion",
        "trillions",
        "billion",
        "billions",
        "million",
        "millions",
        "thousand",
        "thousands",
        "hundred",
        "hundreds",
        "percent",
        "percentage",
        "%",
    ]
    for unit in unit_words:
        cleaned = re.sub(r"\b" + unit + r"\b", "", cleaned)

    # Remove extra whitespace and punctuation
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Check if there's meaningful text left (at least 2 characters)
    has_text = len(cleaned) >= 2

    return has_text, cleaned


def check_text_overlap(gt_text: str, pred_text: str) -> tuple[bool, str]:
    """
    Check if key text elements overlap between ground truth and prediction.

    For hybrid answers like "March 1977", we need to ensure that both:
    1. The numeric part matches (checked elsewhere)
    2. The text part (e.g., "March") appears in both

    Returns: (matches, rationale)

    Examples:
        "March 1977" vs "March 1977" → True
        "March 1977" vs "April 1977" → False (different months)
        "March 1977" vs "1977" → False (missing month in prediction)
    """
    if not gt_text or not pred_text:
        return False, "Empty text in comparison"

    # Get significant text from both
    gt_has_text, gt_cleaned = has_significant_text(gt_text)
    pred_has_text, pred_cleaned = has_significant_text(pred_text)

    if not gt_has_text:
        # GT is purely numeric (e.g., "543 million"), no text check needed
        return True, "GT is purely numeric, text check not required"

    if not pred_has_text:
        # GT has text but prediction doesn't (e.g., GT="March 1977", Pred="1977")
        return False, f"GT has text '{gt_cleaned}' but prediction is purely numeric"

    # Both have text - check if GT text appears in prediction
    # We check substring match because prediction might have more context
    if gt_cleaned in pred_cleaned:
        return True, f"Text overlap: '{gt_cleaned}' found in prediction"

    # Also check reverse (in case GT is longer)
    if pred_cleaned in gt_cleaned:
        return True, f"Text overlap: prediction text '{pred_cleaned}' matches GT"

    # No overlap
    return False, f"Text mismatch: GT='{gt_cleaned}', Pred='{pred_cleaned}'"


def extract_final_answer_from_xml(text: str) -> tuple[str, str | None]:
    if not text:
        return "", None

    matches = list(
        re.finditer(
            r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
    )

    if matches:
        final_answer_match = matches[-1]
        final_answer = final_answer_match.group(1).strip()
        reasoning_before = text[: final_answer_match.start()].strip()
        return final_answer, reasoning_before if reasoning_before else None

    return text, None


def extract_final_answer(text: str) -> str:
    final_answer, _ = extract_final_answer_from_xml(text)
    return final_answer


def fuzzy_match_answer(ground_truth: str, predicted: str, tolerance: float = 0.00) -> tuple[bool, str]:
    """
    Fuzzy match predicted answer against ground truth with robust handling.

    Args:
        ground_truth: The expected answer
        predicted: The model's predicted answer
        tolerance: Numerical tolerance (default 5%)

    Returns:
        (is_correct, rationale)

    Raises:
        ValueError: On any parsing or validation error
    """
    if not ground_truth:
        raise ValueError("Ground truth cannot be empty")
    if not predicted:
        return False, "Predicted answer is empty - marked as incorrect"
    if not 0 <= tolerance <= 1:
        raise ValueError(f"Tolerance must be between 0 and 1, got {tolerance}")

    if "unable to determine" in predicted.lower():
        return False, "Answer contains 'Unable to determine' - marked as incorrect"

    try:
        # Extract numbers with context
        gt_numbers_with_context = extract_numbers_with_context(ground_truth)
        pred_numbers_with_context = extract_numbers_with_context(predicted)
    except Exception as e:
        raise ValueError(f"Failed to extract numbers: {e}") from e

    # Get just the numbers for empty checks
    gt_numbers = [(num, ctx) for num, ctx, _, _ in gt_numbers_with_context]
    pred_numbers = [(num, ctx) for num, ctx, _, _ in pred_numbers_with_context]

    # Case 1: Both have numbers
    if gt_numbers and pred_numbers:
        ordered_list_match = _match_bracketed_numeric_list(
            ground_truth,
            predicted,
            gt_numbers_with_context,
            pred_numbers_with_context,
            tolerance,
        )
        if ordered_list_match is not None:
            return ordered_list_match

        # Check if GT has multiple numbers (list-like answer)
        if len(gt_numbers) > 1:
            # Multi-number answer: ALL GT numbers must appear in prediction
            # Filter out years from predicted numbers first
            pred_non_years = [
                (n, c)
                for n, c in pred_numbers
                if not is_likely_year(n) or any(is_likely_year(g) for g, _ in gt_numbers)
            ]

            matched_gt = []
            unmatched_gt = []

            for gt_val, gt_context in gt_numbers:
                try:
                    gt_base, gt_unit = normalize_number_with_units(gt_val, gt_context)
                except Exception as e:
                    raise ValueError(f"Failed to normalize GT number {gt_val}: {e}") from e

                # Try to find this GT number in predictions
                found_match = False
                for pred_val, pred_context in pred_non_years:
                    try:
                        pred_base, pred_unit = normalize_number_with_units(pred_val, pred_context)
                    except Exception as e:
                        raise ValueError(f"Failed to normalize prediction number {pred_val}: {e}") from e

                    if not units_compatible(gt_unit, pred_unit):
                        continue

                    # Compare base numbers
                    if gt_base == 0:
                        if pred_base == 0:
                            # For multi-number lists, also check text overlap
                            text_matches, _ = check_text_overlap(ground_truth, predicted)
                            if text_matches:
                                found_match = True
                                break
                    else:
                        diff_pct = abs(gt_base - pred_base) / abs(gt_base)
                        if diff_pct <= tolerance:
                            # For multi-number lists, also check text overlap
                            text_matches, _ = check_text_overlap(ground_truth, predicted)
                            if text_matches:
                                found_match = True
                                break

                if found_match:
                    matched_gt.append(gt_val)
                else:
                    unmatched_gt.append(gt_val)

            # All GT numbers must be matched
            if len(matched_gt) == len(gt_numbers):
                return True, f"List match: All {len(gt_numbers)} numbers found in prediction"
            else:
                return (
                    False,
                    f"List mismatch: Found {len(matched_gt)}/{len(gt_numbers)} numbers. Missing: {unmatched_gt}",
                )

        else:
            # Single number answer
            gt_val, gt_context = gt_numbers[0]

            try:
                gt_base, gt_unit = normalize_number_with_units(gt_val, gt_context)
            except Exception as e:
                raise ValueError(f"Failed to normalize GT number: {e}") from e

            # Determine if we should filter year-like numbers from predictions
            # We should NOT filter if:
            # 1. Ground truth itself is a year-like number (e.g., "2003", "1977")
            # 2. Ground truth has text content (e.g., "March 1977" - the 1977 should match)
            gt_has_text, _ = has_significant_text(ground_truth)
            should_filter_years = not (is_likely_year(gt_val) or gt_has_text)

            # Try to find the expected number in the predicted response
            best_match = None
            best_diff = float("inf")
            best_pred_info = None
            unit_mismatches = []

            for pred_val, pred_context in pred_numbers:
                # Skip likely year numbers only if GT is clearly not a year-related answer
                if should_filter_years and is_likely_year(pred_val):
                    continue

                try:
                    pred_base, pred_unit = normalize_number_with_units(pred_val, pred_context)
                except Exception as e:
                    raise ValueError(f"Failed to normalize prediction number: {e}") from e

                if not units_compatible(gt_unit, pred_unit):
                    unit_mismatches.append((pred_base, pred_unit))
                    continue

                # Compare base numbers (after unit normalization)
                # e.g., GT "543 million" → base=543, pred "543" → base=543 ✅
                # e.g., GT "543 million" → base=543, pred "543000000" → base=543000000 ❌

                if gt_base == 0:
                    if pred_base == 0:
                        # Also check text overlap for zero values
                        text_matches, text_rationale = check_text_overlap(ground_truth, predicted)
                        if text_matches:
                            return True, f"Exact match: Found 0 in response. {text_rationale}"
                    continue

                diff_pct = abs(gt_base - pred_base) / abs(gt_base)

                if diff_pct < best_diff:
                    best_diff = diff_pct
                    best_match = pred_base
                    best_pred_info = (pred_base, pred_unit)

                if diff_pct <= tolerance:
                    # Numeric match found - but check if GT has significant text too
                    text_matches, text_rationale = check_text_overlap(ground_truth, predicted)
                    if not text_matches:
                        # Numbers match but text doesn't (e.g., "March 1977" vs "April 1977")
                        continue  # Try other numbers in prediction

                    return (
                        True,
                        f"Numerical match: GT={gt_base} ({gt_unit or 'no unit'}), Pred={pred_base} ({pred_unit or 'no unit'}), Diff={diff_pct * 100:.2f}%. {text_rationale}",
                    )

            # No match found
            if best_match is not None:
                return (
                    False,
                    f"No match: GT={gt_base} ({gt_unit or 'no unit'}), Closest={best_pred_info[0]} ({best_pred_info[1] or 'no unit'}), Diff={best_diff * 100:.2f}%",
                )
            if unit_mismatches:
                pred_units = [unit or "no unit" for _, unit in unit_mismatches[:5]]
                return (
                    False,
                    f"No match: explicit unit mismatch. GT unit={gt_unit or 'no unit'}, Pred units={pred_units}",
                )
            return (
                False,
                f"No valid numbers found in prediction (filtered out years: {[n for n, _ in pred_numbers[:5]]})",
            )

    # Case 2: Text-based comparison (case-insensitive, strip whitespace and quotes)
    # For dates and text, must be EXACT match (case-insensitive)
    gt_clean = ground_truth.strip().lower().strip('"').strip("'")
    pred_clean = predicted.strip().lower().strip('"').strip("'")

    # Strip parenthetical content like (OASI), (FY), etc. to handle abbreviations
    # e.g., "Federal Old-Age and Survivors Insurance (OASI) Trust Fund"
    #    -> "Federal Old-Age and Survivors Insurance Trust Fund"
    gt_clean = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", gt_clean)).strip()
    pred_clean = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", pred_clean)).strip()

    # Check if ground truth appears in prediction
    if gt_clean in pred_clean:
        return True, f"Text match: '{ground_truth}' found in prediction"

    # Check exact match
    if gt_clean == pred_clean:
        return True, "Exact text match"

    # No match
    return False, f"No match found. GT: '{ground_truth[:100]}', Pred: '{predicted[:100]}'"


def _normalize_direct_text_answer(text: str) -> str:
    cleaned = text.strip().lower().strip('"').strip("'")
    cleaned = re.sub(r"\([^)]*\)", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _filter_context_years_for_direct_answer(
    gt_numbers: list[tuple[float, str]],
    pred_numbers: list[tuple[float, str]],
    ground_truth: str,
) -> list[tuple[float, str]]:
    if not gt_numbers:
        return pred_numbers

    gt_has_text, _ = has_significant_text(ground_truth)
    if len(gt_numbers) == 1:
        gt_val, _ = gt_numbers[0]
        should_filter_years = not (is_likely_year(gt_val) or gt_has_text)
    else:
        should_filter_years = not any(is_likely_year(gt_val) for gt_val, _ in gt_numbers)

    if not should_filter_years:
        return pred_numbers

    return [(pred_val, pred_context) for pred_val, pred_context in pred_numbers if not is_likely_year(pred_val)]


_INLINE_MARKUP_RE = re.compile(r"<[^>]*>")


def _is_direct_answer_only(ground_truth: str, predicted: str) -> tuple[bool, str]:
    """
    Guard that prediction is a single, direct answer of the same shape as GT.

    Policy:
        - Empty prediction -> fail
        - Multiple non-empty lines -> fail
        - Predictions longer than 250 chars -> fail
        - Numeric GT -> prediction must contain the same count of numeric values
        - Purely numeric GT -> prediction must not contain prose after removing
          numbers and unit words
        - Text/date GT (no numbers) -> normalized prediction must equal normalized GT

    Returns:
        (ok, rationale)
    """
    predicted = predicted.strip()
    if not predicted:
        return False, "Predicted answer is empty"

    nonempty_lines = [line for line in predicted.splitlines() if line.strip()]
    if len(nonempty_lines) > 1:
        return False, "Predicted answer spans multiple non-empty lines"

    if len(predicted) > 250:
        return False, "Predicted answer is too long to be a direct answer"

    gt_numbers_with_context = extract_numbers_with_context(ground_truth)
    pred_numbers_with_context = extract_numbers_with_context(predicted)

    gt_numbers = [(num, ctx) for num, ctx, _, _ in gt_numbers_with_context]
    pred_numbers = [(num, ctx) for num, ctx, _, _ in pred_numbers_with_context]

    if gt_numbers:
        pred_numbers = _filter_context_years_for_direct_answer(gt_numbers, pred_numbers, ground_truth)
        if len(pred_numbers) != len(gt_numbers):
            gt_list_numbers = _ground_truth_bracketed_numeric_list(ground_truth, gt_numbers_with_context)
            list_arity_ok = gt_list_numbers is not None and bool(
                _segment_numeric_list_values(predicted, len(gt_list_numbers))
            )
            if not list_arity_ok:
                return (
                    False,
                    "Predicted answer must contain exactly the expected answer numbers",
                )

        if _INLINE_MARKUP_RE.search(predicted):
            return False, "Predicted answer contains markup outside the answer value"

        return True, "Direct answer only"

    if pred_numbers:
        return False, "Prediction contains numbers but ground truth is text-only"

    if _normalize_direct_text_answer(ground_truth) != _normalize_direct_text_answer(predicted):
        return False, "Text answer must match the expected answer text exactly"

    return True, "Direct answer only"


def score_answer(ground_truth: str, predicted: str, tolerance: float = 0.00) -> float:
    """
    Score the answer using robust fuzzy matching.
    """
    try:
        predicted, _ = extract_final_answer_from_xml(predicted)
    except Exception:
        return 0.0

    try:
        ok, _ = _is_direct_answer_only(ground_truth, predicted)
        if not ok:
            return 0.0

        is_correct, _ = fuzzy_match_answer(ground_truth, predicted, tolerance)
    except Exception:
        return 0.0

    return 1.0 if is_correct else 0.0
