# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import re
from pathlib import Path
from typing import ClassVar

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.chembench.task_data import TaskData


THINK_BLOCK = re.compile(r"<(think|thinking)\b[^>]*>.*?(?:</\1\s*>|$)", re.IGNORECASE | re.DOTALL)
THINK_END = re.compile(r"</(?:think|thinking)\s*>", re.IGNORECASE)
ANSWER_TAG = re.compile(r"\[(ANSWER|ANS)\](.*?)\[/\1\]|<(ANSWER|ANS)>(.*?)</\3>", re.IGNORECASE | re.DOTALL)
ANSWER_MARKER = re.compile(r"[\[<]/?(?:ANSWER|ANS)\b", re.IGNORECASE)
LETTERS = re.compile(r"[A-Z](?:(?:\s*,\s*|\s+)[A-Z])*")
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
LETTER_LIST = re.compile(r"\b[A-Z]\b(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s+)\b[A-Z]\b)*")
# Consume connected arithmetic as one candidate so an unsupported expression cannot
# silently turn into its exponent or denominator.
MATH_OPERATOR = r"(?:\\times|\\cdot|\*\*|[*/^×÷+−-])"
NUMERIC_EXPRESSION = re.compile(
    rf"(?<![\w.,]){NUMBER.pattern}(?:\s*{MATH_OPERATOR}\s*\{{?{NUMBER.pattern}\}}?)*(?![\w.,])"
)
FRACTION = re.compile(rf"({NUMBER.pattern})\s*[/÷]\s*({NUMBER.pattern})")
POWER_OF_TEN = re.compile(
    rf"(?:(?P<coefficient>{NUMBER.pattern})\s*(?:\*|×|\\times|\\cdot)\s*)?"
    r"10\s*(?:\^|\*\*)\s*(?:\{(?P<braced>[+-]?[0-9]+)\}|(?P<plain>[+-]?[0-9]+))"
)


def _parse_number(candidate: str) -> float | None:
    try:
        if NUMBER.fullmatch(candidate):
            number = float(candidate)
        elif match := FRACTION.fullmatch(candidate):
            numerator, denominator = float(match.group(1)), float(match.group(2))
            if not math.isfinite(numerator) or not math.isfinite(denominator):
                return None
            number = numerator / denominator
        elif match := POWER_OF_TEN.fullmatch(candidate):
            exponent = int(match.group("braced") or match.group("plain"))
            number = float(match.group("coefficient") or "1") * 10.0**exponent
        else:
            return None
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) else None


def extract_answer(text: str, question_type: str) -> str | float | None:
    """Read the last tagged answer, or an entirely bare answer, outside reasoning blocks.

    Exact parsing preserves multi-select lists. If tagged content contains extra prose,
    fall back to its last uppercase letter list or numeric expression. Letter lists may use
    commas, whitespace, or "and", including inside surrounding punctuation. Untagged
    prose is not searched, and numeric parsing supports fractions and powers of ten.
    """
    text = THINK_BLOCK.sub("", text)
    # Some providers omit the opening reasoning tag from the generated text.
    text = THINK_END.split(text)[-1].strip()
    matches = list(ANSWER_TAG.finditer(text))
    if matches:
        match = matches[-1]
        candidate = (match.group(2) if match.group(1) else match.group(4)).strip()
    elif ANSWER_MARKER.search(text):
        return None
    else:
        candidate = text

    if question_type == "mcq":
        if LETTERS.fullmatch(candidate):
            return ", ".join(sorted(set(re.findall(r"[A-Z]", candidate))))
        lists = LETTER_LIST.findall(candidate) if matches else []
        return ", ".join(sorted(set(re.findall(r"[A-Z]", lists[-1])))) if lists else None
    # Bare responses must be entirely numeric; only tagged answers allow prose.
    if not matches:
        return _parse_number(candidate)
    expressions = list(NUMERIC_EXPRESSION.finditer(candidate))
    if not expressions:
        return None
    expression = expressions[-1]
    before, after = candidate[: expression.start()].rstrip(), candidate[expression.end() :].lstrip()
    # Reject a partial match adjacent to arithmetic, including malformed exponents.
    if re.search(r"(?:[*/^×÷+−-]|\\times|\\cdot)[\s({]*$", before) or re.match(MATH_OPERATOR, after):
        return None
    if before.endswith("{") or after.startswith("}") or re.search(r"\b(?:sqrt|sin|cos|tan|log|ln|exp)\s*\($", before):
        return None
    return _parse_number(expression.group(0))


class ChembenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class ChembenchVerifyRequest(BaseVerifyRequest):
    verifier_metadata: TaskData


class ChembenchVerifyResponse(BaseVerifyResponse, ChembenchVerifyRequest):
    predicted_answer: str | float | None
    no_answer: bool


class ChembenchVerifier:
    async def verify(self, body: ChembenchVerifyRequest) -> ChembenchVerifyResponse:
        metadata = body.verifier_metadata
        predicted = extract_answer(body.response.output_text or "", metadata.question_type)
        if metadata.question_type == "mcq":
            expected = ", ".join(sorted(set(re.findall(r"[A-Z]", metadata.expected_answer))))
            correct = predicted is not None and predicted == expected
        else:
            expected = float(metadata.expected_answer)
            # Match upstream prompter._calculate_metrics, including signed/zero targets.
            tolerance = metadata.relative_tolerance
            if tolerance is None:
                tolerance = 0.01 * expected
            correct = predicted is not None and abs(predicted - expected) < tolerance
        return ChembenchVerifyResponse(
            **body.model_dump(),
            reward=float(correct),
            predicted_answer=predicted,
            no_answer=predicted is None,
        )


class ChembenchResourcesServer(ChembenchVerifier, SimpleResourcesServer):
    ray_enabled = False

    config: ChembenchResourcesServerConfig


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=ChembenchVerifier,
    request_model=ChembenchVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    ChembenchResourcesServer.run_webserver()
