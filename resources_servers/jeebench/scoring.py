# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Scoring adapted from JEEBench (MIT), Copyright (c) 2023 DAIR Group, IIT Delhi.
# See NOTICE for the upstream license; NVIDIA modifications are Apache-2.0.
"""Dependency-light JEEBench answer normalization and published scoring."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any, Literal

from resources_servers.jeebench.extraction import extract_formatted_answer


QuestionType = Literal["MCQ", "MCQ(multiple)", "Integer", "Numeric"]
QUESTION_TYPES = {"MCQ", "MCQ(multiple)", "Integer", "Numeric"}
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def compute_score(gold: str, response: str, question_type: QuestionType) -> float:
    """Reproduce upstream compute_metrics.compute_score on extracted answers."""
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"Unknown JEEBench question type: {question_type!r}")
    if question_type in {"MCQ", "MCQ(multiple)"}:
        expected = {letter for letter in "ABCD" if letter in gold}
        predicted = {letter for letter in "ABCD" if letter in response}
        if expected == predicted:
            return 1.0
        if question_type == "MCQ(multiple)" and not predicted - expected:
            return 0.25 * len(predicted)
        return 0.0
    if response == "None":
        return 0.0
    return float(abs(float(gold) - float(response)) <= 0.01)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def visible_answer(response: Any) -> str:
    """Return visible final-channel text from a native response model or JSON mapping."""
    text = "\n".join(
        _field(part, "text", "")
        for item in _field(response, "output", [])
        if _field(item, "type") == "message"
        for part in _field(item, "content", [])
        if _field(part, "type") == "output_text"
    )
    text = re.split(r"</(?:think|thinking)>", text, flags=re.IGNORECASE)[-1]
    return re.split(r"<(?:think|thinking)>", text, flags=re.IGNORECASE)[0].strip()


def _unwrap(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().strip("$`*_ ")
    while match := re.fullmatch(r"\\(?:text|mathrm|operatorname)\{(.*)\}", text, flags=re.DOTALL):
        text = match[1].strip()
    return text.replace("−", "-")


def normalize_answer(text: str, question_type: QuestionType) -> str | None:
    """Validate an extracted value and reject prose or type-incompatible answers."""
    text = _unwrap(text)
    if text.lower() == "none":
        return "None"
    if question_type in {"MCQ", "MCQ(multiple)"}:
        text = re.sub(r"\band\b", ",", text, flags=re.IGNORECASE)
        text = text.strip().strip(".")
        if not re.fullmatch(r"[A-Da-d\s,;(){}\[\]&]+", text):
            return None
        options = sorted(set(re.findall(r"[A-D]", text.upper())))
        if not options or (question_type == "MCQ" and len(options) != 1):
            return None
        return "".join(options)
    if not NUMBER.fullmatch(text):
        return None
    value = float(text)
    if not math.isfinite(value):
        return None
    if question_type == "Integer" and (value < 0 or not value.is_integer()):
        return None
    return str(int(value)) if question_type == "Integer" else text


def extract_answer(text: str, question_type: QuestionType, *, question: str = "") -> tuple[str, str]:
    """Extract one explicit answer without access to the reference answer."""
    return extract_formatted_answer(text, question_type, normalize_answer, question=question)
