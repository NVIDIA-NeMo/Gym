# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, Optional, Tuple

from .. import validator as base_validator
from ..data_loader import EXPECTED_ARGUMENTS


def check_relation(count, relation: str, expected: int) -> Tuple[bool, Optional[str]]:
    """
    Check if a count satisfies a relation against an expected value.

    Args:
        count: The actual count to check (must be numeric)
        relation: One of "at least", "equal to", "less than"
        expected: The expected value to compare against (must be numeric)

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid,
        or an error string if validation fails.
    """
    # Validate count is numeric
    if not isinstance(count, (int, float)):
        return (False, f"Invalid count type: expected numeric, got {type(count).__name__}.")

    # Validate expected is numeric
    if not isinstance(expected, (int, float)):
        return (False, f"Invalid expected value type: expected numeric, got {type(expected).__name__}.")

    # Validate relation
    if relation == "at least":
        return (count >= expected, None)
    elif relation == "equal to":
        return (count == expected, None)
    elif relation == "less than":
        return (count < expected, None)
    else:
        return (False, f"Invalid relation: '{relation}'. Must be 'at least', 'equal to', or 'less than'.")


def _normalize_casefold(text: str, lang: str) -> str:
    """Normalize and casefold text using the base validator's language strategy."""
    return base_validator._get_strategy(lang).casefold(text)


CHINESE_COMMA_SET = {",", "，", "、"}
CHINESE_PERIOD_SET = {".", "。", "．"}
QUESTION_EXCLAIM_PATTERN = re.compile(r"[?？!！]")


def _relation_ok(count: int, relation: str, target: int) -> bool:
    if relation in {"equal to", "=="}:
        return count == target
    if relation in {"at least", ">="}:
        return count >= target
    if relation in {"less than", "<"}:
        return count < target
    raise ValueError("Invalid relation. Use 'equal to', 'at least', or 'less than'.")


def _keyword_count(text: str, keyword: str) -> int:
    kw = (keyword or "").strip()
    if not kw:
        return 0
    normalized_text = _normalize_casefold(text, "zh")
    normalized_kw = _normalize_casefold(kw, "zh")
    return normalized_text.count(normalized_kw)


def _count_placeholders(text: str) -> int:
    """
    Counts placeholder tokens of the form {TOKEN} or [TOKEN].
    """
    curly = re.findall(r"\{[^{}]+\}", text)
    square = re.findall(r"\[[^\[\]]+\]", text)
    return len(curly) + len(square)


def validate_instruction(
    response: str, inst_type: str, kwargs: Dict[str, Any], all_instructions: Dict = None
) -> Tuple[bool, str]:
    if inst_type not in EXPECTED_ARGUMENTS:
        return False, f"Instruction '{inst_type}' is not supported for language 'zh'."

    response = response.strip()

    if inst_type == "detectable_format:json_format":
        return base_validator.validate_instruction_generic(
            response, inst_type, kwargs, all_instructions, language="zh"
        )

    if inst_type == "detectable_content:number_placeholders":
        count = _count_placeholders(response)
        rel, val = kwargs["relation"], kwargs["num_placeholders"]
        valid = _relation_ok(count, rel, val)
        return valid, ("No error" if valid else f"Expected {rel} {val} placeholders, found {count}.")

    if inst_type == "detectable_format:multiple_sections":
        splitter = (kwargs.get("section_splitter") or "").strip()
        rel = kwargs.get("relation")
        val = kwargs.get("num_sections")

        if not splitter:
            return False, "section_splitter cannot be empty."

        header_re = re.compile(
            rf"^\s*(?:#+\s*)?{re.escape(splitter)}\s+\d+\b",
            re.MULTILINE,
        )
        sections = header_re.findall(response)
        count = len(sections)

        if count == 0 and splitter == "---":
            hr_matches = re.findall(r"^\s*---+\s*$", response, re.MULTILINE)
            count = len(hr_matches)

        if count == 0 and splitter.startswith("<<") and splitter.endswith(">>"):
            generic_marks = re.findall(r"^\s*<<[^>]+>>\s*$", response, re.MULTILINE)
            count = len(generic_marks)

        if count == 0:
            parts = [p for p in re.split(r"\n\s*\n", response) if p.strip()]
            if splitter and len(parts) > 1 and any(splitter in p for p in parts):
                count = len(parts)

        valid, err = check_relation(count, rel, val)
        if err is not None:
            return False, err
        return valid, ("No error" if valid else f"Expected {rel} {val} sections, found {count}.")

    if inst_type == "detectable_format:number_bullet_lists":
        bullet_pattern = re.compile(r"^[ \t]*[*\-+]\s+", re.MULTILINE)
        numbered_pattern = re.compile(r"^[ \t]*\d+[.)]\s+", re.MULTILINE)
        count = len(bullet_pattern.findall(response)) + len(numbered_pattern.findall(response))
        rel, val = kwargs["relation"], kwargs["num_bullets"]
        valid = _relation_ok(count, rel, val)
        return valid, ("No error" if valid else f"Expected {rel} {val} bullet points, found {count}.")

    if inst_type == "detectable_format:title":
        first_line = response.splitlines()[0].strip() if response.splitlines() else ""
        is_marker = first_line.startswith("<<") and first_line.endswith(">>")
        is_heading = first_line.startswith("#")
        is_standalone = bool(first_line) and (len(response.splitlines()) <= 2)
        if is_marker or is_heading or is_standalone:
            return True, "No error"
        return False, "Title not detected on first line (accepts << >>, Markdown heading, or first-line title)."

    if inst_type == "keywords:existence":
        missing = [kw for kw in kwargs["keywords"] if _keyword_count(response, kw) == 0]
        return (not missing, "No error" if not missing else f"Missing keyword(s): {missing}")

    if inst_type == "keywords:frequency":
        count = _keyword_count(response, kwargs["keyword"])
        relation, target = kwargs["relation"], kwargs["frequency"]
        valid = _relation_ok(count, relation, target)
        return valid, (
            "No error" if valid else f"Expected {relation} {target} of '{kwargs['keyword']}', found {count}."
        )

    if inst_type == "keywords:forbidden_words":
        present = [kw for kw in kwargs["forbidden_words"] if _keyword_count(response, kw) > 0]
        return (not present, "No error" if not present else f"Forbidden words found: {present}")

    if inst_type == "keywords:letter_frequency":
        letter = kwargs["letter"]
        count = _keyword_count(response, letter)
        relation, target = kwargs["let_relation"], kwargs["let_frequency"]
        valid = _relation_ok(count, relation, target)
        return valid, ("No error" if valid else f"Expected {relation} {target} '{letter}', found {count}.")

    if inst_type == "punctuation:no_comma":
        has_comma = any(p in response for p in CHINESE_COMMA_SET)
        return (not has_comma, "No error" if not has_comma else "Commas found in response.")

    if inst_type == "punctuation:no_period":
        has_period = any(p in response for p in CHINESE_PERIOD_SET)
        return (not has_period, "No error" if not has_period else "Periods found in response.")

    if inst_type == "punctuation:question_exclaim":
        count = len(QUESTION_EXCLAIM_PATTERN.findall(response))
        relation, target = kwargs["relation"], kwargs["num_marks"]
        valid = _relation_ok(count, relation, target)
        return valid, ("No error" if valid else f"Found {count} marks. Expected {relation} {target}.")

    if inst_type == "punctuation:frequency":
        punct = kwargs.get("punctuation", "")
        relation, target = kwargs.get("relation", "at least"), kwargs.get("frequency", 0)
        count = response.count(punct)
        valid = _relation_ok(count, relation, target)
        return valid, (
            "No error" if valid else f"Found {count} occurrences of '{punct}'. Expected {relation} {target}."
        )

    if inst_type == "punctuation:end_rule":
        allowed = kwargs["allowed"]
        stripped = response.rstrip()
        if not stripped:
            return False, "Empty response."
        m = re.search(r"[。．.!?！？?!；;：:，,]$", stripped)
        if not m:
            return False, "Required ending punctuation not found."
        last = m.group(0)
        if last not in allowed:
            return False, f"Ending punctuation '{last}' not allowed. Allowed: {allowed}"
        return True, "No error."

    if inst_type == "startend:wrap_checker":
        wrap = kwargs["wrap_phrase"]
        WRAP_PAIRS = [
            ("《", "》"),
            ("〈", "〉"),
            ("「", "」"),
            ("『", "』"),
            ("【", "】"),
            ("[", "]"),
            ("(", ")"),
            ("（", "）"),
            ("{", "}"),
            ("<", ">"),
        ]
        if len(wrap) == 1:
            return (
                response.startswith(wrap) and response.endswith(wrap),
                "No error" if response.startswith(wrap) and response.endswith(wrap) else f"Not wrapped with: {wrap}",
            )
        if len(wrap) == 2:
            left, right = wrap[0], wrap[1]
            if response.startswith(left) and response.endswith(right):
                return True, "No error"
        for left_delim, right_delim in WRAP_PAIRS:
            if response.startswith(left_delim) and response.endswith(right_delim):
                return True, "No error"
        return False, f"Not wrapped with: {wrap}"

    if inst_type == "startend:end_checker":
        required = kwargs["end_phrase"].strip()
        if not required:
            return False, "End phrase is empty."
        normalized = response.rstrip()
        if normalized.endswith(required):
            return True, "No error"
        stripped = re.sub(r"[。．.?!！？…\s]+$", "", normalized)
        if stripped.endswith(required):
            return True, "No error"
        return False, f"End phrase mismatch: expected to end with '{required}'."

    if inst_type == "length_constraints:sentence_length":
        max_words = kwargs["max_words"]
        sentences = re.split(r"[。．.!?！？]+", response)
        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue
            char_count = len(re.sub(r"\s+", "", s))
            if char_count > max_words:
                return False, f"Sentence length {char_count} exceeds max {max_words}."
        return True, "No error."

    if inst_type == "startend:quotation":
        quote_pairs = [('"', '"'), ("“", "”"), ("「", "」"), ("『", "』"), ("《", "》"), ("〈", "〉")]
        for start, end in quote_pairs:
            if response.startswith(start) and response.endswith(end):
                return True, "No error"
        return False, "Response not wrapped in accepted quotation marks."

    return base_validator.validate_instruction_generic(response, inst_type, kwargs, all_instructions, language="zh")
