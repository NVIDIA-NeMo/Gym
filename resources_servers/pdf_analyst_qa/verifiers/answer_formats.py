# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse the answer format requested by each PDF Analyst QA task."""

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


type AnswerFormat = Literal[
    "bare_value",
    "value_with_unit",
    "tagged",
    "value_with_citation",
    "json_object",
    "structured_json",
    "bracket_list",
]


@dataclass(frozen=True)
class ParsedAnswer:
    value: str
    unit: str | None = None
    citation_document: str | None = None
    citation_page: int | None = None
    citations: tuple[tuple[str, int], ...] = ()


def _split_unit(value: str, unit: str | None) -> tuple[str, str | None]:
    stripped = value.strip()
    if unit is None:
        return stripped, None
    if unit == "percent" and stripped.endswith("%"):
        return stripped[:-1].strip(), unit
    # Currency symbols are units, not universal numerical punctuation. Strip
    # one only when it names the explicitly requested (unscaled) currency.
    currency = {
        "dollars": "$",
        "US dollars": "$",
        "USD": "$",
        "euros": "€",
        "EUR": "€",
        "pounds sterling": "£",
        "GBP": "£",
        "yen": "¥",
        "JPY": "¥",
    }.get(unit)
    if currency is not None:
        match = re.fullmatch(rf"([+-]?)\s*{re.escape(currency)}\s*(.+)", stripped)
        if match is not None:
            return match.group(1) + match.group(2), unit
    suffixes = [unit]
    if unit in {"thousands of dollars", "millions of dollars", "billions of dollars"}:
        scale = unit.split(" ", 1)[0]
        suffixes.insert(0, f"{scale.removesuffix('s')} dollars")
    # Known word units tolerate capitalization; symbols such as mW/MW do not.
    word_unit = unit in {"percent", "thousands of dollars", "millions of dollars", "billions of dollars"}
    for suffix in suffixes:
        if stripped.endswith(suffix) or word_unit and stripped.casefold().endswith(suffix.casefold()):
            return stripped[: -len(suffix)].strip(), unit
    raise ValueError(f"answer omits requested unit {unit!r}")


def _without_sentence_period(value: str, expected_unit: str | None) -> str:
    """Drop sentence punctuation after a required unit; preserve transcribed text."""

    if expected_unit is None or not value.endswith("."):
        return value
    return value[:-1].rstrip()


def parse_answer(
    response: str,
    *,
    answer_format: AnswerFormat,
    expected_unit: str | None,
) -> ParsedAnswer:
    stripped = response.strip()
    if not stripped:
        raise ValueError("answer is blank")
    if answer_format == "bracket_list":
        parsed = json.loads(stripped, parse_float=Decimal)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("bracket-list answer must be a nonempty list")
        pending = list(parsed)
        while pending:
            item = pending.pop()
            if isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, dict) or item is None or isinstance(item, bool):
                raise ValueError("bracket-list slots must be labels or numbers")
        return ParsedAnswer(value=stripped)
    if answer_format == "structured_json":
        parsed = json.loads(stripped, parse_float=Decimal)
        if not isinstance(parsed, (dict, list)) or not parsed:
            raise ValueError("structured answer must be a nonempty object or list")
        return ParsedAnswer(value=stripped)
    if answer_format == "bare_value":
        if (
            "\n" in stripped
            or stripped.startswith(("<", "{", "```"))
            or re.search(r"\s+—\s+.+,\s+PDF page\s+\d+\.?$", stripped)
        ):
            raise ValueError("bare-value answer contains a wrapper")
        if expected_unit is not None:
            try:
                _split_unit(stripped, expected_unit)
            except ValueError:
                pass
            else:
                raise ValueError("bare-value answer restates the unit")
        return ParsedAnswer(value=stripped)
    if answer_format == "value_with_unit":
        value, unit = _split_unit(_without_sentence_period(stripped, expected_unit), expected_unit)
        return ParsedAnswer(value=value, unit=unit)
    if answer_format == "tagged":
        match = re.fullmatch(
            r"<REASONING>.*?</REASONING>\s*<FINAL_ANSWER>(.*?)</FINAL_ANSWER>",
            stripped,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError("tagged answer does not contain the required XML elements")
        value, unit = _split_unit(match.group(1), expected_unit)
        return ParsedAnswer(value=value, unit=unit)
    if answer_format == "value_with_citation":
        parts = stripped.rsplit(" — ", 1)
        if len(parts) != 2:
            raise ValueError("citation answer does not contain a source suffix")
        value, unit = _split_unit(parts[0], expected_unit)
        citations: list[tuple[str, int]] = []
        for item in parts[1].removesuffix(".").split("; "):
            citation = re.fullmatch(r"(.+?),\s+PDF page\s+(\d+)", item)
            if citation is None:
                raise ValueError("citation answer does not end with document names and PDF pages")
            citations.append((citation.group(1), int(citation.group(2))))
        return ParsedAnswer(
            value=value,
            unit=unit,
            citation_document=citations[0][0],
            citation_page=citations[0][1],
            citations=tuple(citations),
        )
    if answer_format == "json_object":
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict) or set(parsed) != {"answer", "unit"}:
            raise ValueError("JSON answer must contain exactly answer and unit")
        if not isinstance(parsed["answer"], str) or parsed["unit"] is not None and not isinstance(parsed["unit"], str):
            raise ValueError("JSON answer must be text and unit must be text or null")
        if parsed["unit"] != expected_unit:
            raise ValueError("JSON answer unit does not match the requested unit")
        return ParsedAnswer(value=parsed["answer"], unit=parsed["unit"])
    raise ValueError(f"unknown requested answer format: {answer_format}")
