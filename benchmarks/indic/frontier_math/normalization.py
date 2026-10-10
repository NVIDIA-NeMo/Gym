# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative answer-format recovery for auxiliary normalized accuracy."""

import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


GRADER_PATH = Path(__file__).resolve().parents[3] / "resources_servers/frontiermath/grading.py"
RECOVERABLE_STATUSES = {"missing_answer", "invalid_expression"}


@dataclass(frozen=True)
class NormalizedGrade:
    reward: float
    recovered_answer: str | None = None
    recovery_method: str | None = None


def _response_text(row: dict) -> str:
    """Return visible assistant text from a serialized Responses API response."""
    response = row.get("response") or {}
    chunks = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "".join(chunks)


def _last_complete_box(text: str) -> str | None:
    """Find the last complete box outside hidden reasoning tags."""
    text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.split(r"<(?:think|thinking)>", text, maxsplit=1, flags=re.IGNORECASE)[0]
    answers = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        for index in range(match.end(), len(text)):
            depth += (text[index] == "{") - (text[index] == "}")
            if depth == 0:
                answer = text[match.end() : index].strip()
                if answer:
                    answers.append(answer)
                break
    return answers[-1] if answers else None


def _top_level_rhs(text: str) -> str | None:
    """Return the final right-hand side when an answer is written as an equality."""
    depth = 0
    last_equals = None
    for index, character in enumerate(text):
        depth += (character == "{") - (character == "}")
        if character == "=" and depth == 0:
            last_equals = index
    if last_equals is None:
        return None
    answer = text[last_equals + 1 :].strip()
    return answer or None


def _normalize_latex_format(text: str) -> str:
    """Remove presentation-only LaTeX that commonly prevents exact parsing."""
    normalized = re.sub(r"\\(?:dfrac|tfrac)\b", r"\\frac", text)
    normalized = re.sub(r"\\(?:displaystyle|left|right|Biggl|Biggr|biggl|biggr|Bigl|Bigr|bigl|bigr)\b", "", normalized)
    normalized = re.sub(r"\\[,;!:]", "", normalized)
    return re.sub(r"\s+", "", normalized)


def recovery_candidates(row: dict) -> Iterator[tuple[str, str]]:
    """Yield gold-independent formatting recoveries in deterministic order."""
    extracted = row.get("extracted_answer")
    complete_box = _last_complete_box(_response_text(row))
    bases = []
    if complete_box:
        bases.append(("last_complete_box", complete_box))
    if isinstance(extracted, str) and extracted.strip():
        rhs = _top_level_rhs(extracted)
        if rhs:
            bases.append(("top_level_rhs", rhs))
        bases.append(("latex_format", extracted))

    seen = {extracted} if isinstance(extracted, str) else set()
    for method, candidate in bases:
        variants = [(method, candidate)]
        formatted = _normalize_latex_format(candidate)
        if formatted != candidate:
            formatted_method = "latex_format" if method == "latex_format" else f"{method}+latex_format"
            variants.append((formatted_method, formatted))
        for variant_method, variant in variants:
            if variant and variant not in seen:
                seen.add(variant)
                yield variant_method, variant


def _grade_candidate(row: dict, candidate: str) -> bool:
    """Run the exact verifier in a bounded subprocess."""
    candidate = unicodedata.normalize("NFKC", candidate).replace("−", "-")
    candidate = "".join(str(unicodedata.decimal(char)) if char.isdecimal() else char for char in candidate)
    if row["answer_type"] == "integer" and re.fullmatch(r"[+-]?\d+", candidate):
        return int(candidate) == int(row["expected_answer"])
    request = {
        "expected_answer": row["expected_answer"],
        "answer_type": row["answer_type"],
        "generated_answer": rf"\boxed{{{candidate}}}",
    }
    try:
        process = subprocess.run(
            [sys.executable, str(GRADER_PATH)],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        result = json.loads(process.stdout) if process.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return False
    return result.get("reward") == 1.0


def normalized_grade(row: dict) -> NormalizedGrade:
    """Keep strict credit and recover only exact answers rejected for formatting."""
    strict_reward = float(row["reward"])
    if strict_reward == 1.0 or row["grading_status"] not in RECOVERABLE_STATUSES:
        return NormalizedGrade(strict_reward)
    for method, candidate in recovery_candidates(row):
        if _grade_candidate(row, candidate):
            return NormalizedGrade(1.0, candidate, method)
    return NormalizedGrade(0.0)
