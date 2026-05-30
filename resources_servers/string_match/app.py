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
import unicodedata
from typing import Any, Literal, Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class StringMatchResourcesServerConfig(BaseResourcesServerConfig):
    pass


class StringMatchRunRequest(BaseRunRequest):
    expected_answer: str
    extraction_mode: Literal[
        "boxed",
        "final_answer",
        "last_line",
        "full_response",
    ] = "final_answer"
    case_sensitive: bool = False
    metadata: Optional[dict[str, Any]] = None


class StringMatchVerifyRequest(StringMatchRunRequest, BaseVerifyRequest):
    pass


class StringMatchVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]


BOXED_PATTERN = re.compile(r"\\boxed\{\s*(.*?)\s*\}", re.S)
LATEX_TEXT_WRAP = re.compile(r"\\text\{\s*(.*?)\s*\}", re.S)
FINAL_ANSWER_PATTERN = re.compile(
    r"(?i)(?:final\s+answer|answer)\s*[:：]\s*(.+)",
)


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


def _strip_latex_wrappers(s: str) -> str:
    while True:
        m = LATEX_TEXT_WRAP.fullmatch(s)
        if not m:
            break
        s = m.group(1)
    return s


def _extract_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content."""
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    inner = matches[-1].strip()
    return _strip_latex_wrappers(inner).strip()


def _extract_final_answer(text: str) -> Optional[str]:
    """Extract the last 'Final answer: ...' or 'Answer: ...' value.

    Captures everything after the colon until a newline, period followed
    by whitespace/end, or end of string.
    """
    matches = FINAL_ANSWER_PATTERN.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip()
    raw = re.split(r"\.\s*$", raw)[0].strip()
    raw = raw.rstrip(".")
    return raw


def _extract_last_line(text: str) -> Optional[str]:
    """Return the last non-empty line of the response."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return None


def _extract_answer(text: str, mode: str) -> Optional[str]:
    if mode == "boxed":
        return _extract_boxed(text)
    elif mode == "final_answer":
        result = _extract_final_answer(text)
        if result is None:
            result = _extract_boxed(text)
        return result
    elif mode == "last_line":
        return _extract_last_line(text)
    elif mode == "full_response":
        return text.strip() if text.strip() else None
    return None


def _grade_string_match(gt_answer: str, pred_answer: str) -> float:
    """NeMoRL string-match verifier without format reward."""
    pred_answer = unicodedata.normalize("NFKC", pred_answer)
    gt_answer = unicodedata.normalize("NFKC", gt_answer)
    pred_answer = _strip_outer_quotes(pred_answer).lower()
    gt_answer = _strip_outer_quotes(gt_answer).lower()
    pred_answer = _strip_trailing_punctuation(pred_answer)
    gt_answer = _strip_trailing_punctuation(gt_answer)

    if pred_answer == gt_answer:
        return 1.0
    if _normalize_float(pred_answer) == _normalize_float(gt_answer):
        return 1.0
    if _normalize_numbers(pred_answer) == _normalize_numbers(gt_answer):
        return 1.0
    if _normalize_latex(pred_answer) == _normalize_latex(gt_answer):
        return 1.0
    if _normalize_lists(pred_answer) == _normalize_lists(gt_answer):
        return 1.0
    if _normalize_states(pred_answer) == _normalize_states(gt_answer):
        return 1.0
    if _soft_numeric_grade(gt_answer, pred_answer) > 0.98:
        return 0.98

    pred_stripped = _MATH_UNITS_RE.sub(
        "", _normalize_unicode_math(_normalize_latex(pred_answer))
    ).strip()
    gt_stripped = _MATH_UNITS_RE.sub(
        "", _normalize_unicode_math(_normalize_latex(gt_answer))
    ).strip()
    if pred_stripped == gt_stripped:
        return 1.0
    if _soft_numeric_grade(gt_stripped, pred_stripped) > 0.98:
        return 0.98
    return 0.0


def _answers_match(extracted: str, expected: str, case_sensitive: bool) -> float:
    if case_sensitive:
        return 1.0 if extracted.strip() == expected.strip() else 0.0
    return _grade_string_match(expected, extracted)


def _strip_outer_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _strip_trailing_punctuation(text: str) -> str:
    stripped = text.rstrip(".!?")
    return stripped or text


def _normalize_float(text: str) -> str:
    cleaned = text.replace("\\%", "").replace("\\$", "").replace("$", "").strip()
    try:
        return str(float(cleaned))
    except ValueError:
        return text


def _normalize_numbers(text: str) -> str:
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    text = text.replace("\\%", "%")
    return text


def _normalize_lists(text: str) -> str:
    orig_text = text
    orig_len = len(text)
    text = text.replace(",", " ").replace(";", " ")
    text = text.replace("and", " ").replace("or", " ")
    if len(text) < 0.9 * orig_len:
        return orig_text
    text = " ".join(text.split())
    return text


def _normalize_latex(text: str) -> str:
    for cmd in (
        "\\boxed",
        "\\text",
        "\\textbf",
        "\\textit",
        "\\texttt",
        "\\mathrm",
        "\\mathbf",
        "\\mathit",
        "\\mathsf",
        "\\mathbb",
        "\\mathcal",
        "\\emph",
        "\\url",
    ):
        text = _remove_latex_command(cmd, text)
    for token in ("\\(", "\\)", "\\[", "\\]"):
        text = text.replace(token, "")
    text = text.replace("$", "")
    text = " ".join(text.split())
    return text


def _remove_latex_command(cmd: str, text: str) -> str:
    assert cmd.startswith("\\"), f"command must start with \\: {cmd}"
    while cmd + "{" in text:
        prefix, suffix = text.split(cmd + "{", 1)
        depth = 1
        for i, char in enumerate(suffix):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0:
                text = prefix + suffix[:i] + suffix[i + 1 :]
                break
        if depth != 0:
            text = prefix + suffix
    return text


def _normalize_states(text: str) -> str:
    text = (
        text.replace("district of columbia", "DC")
        .replace("new hampshire", "NH")
        .replace("new jersey", "NJ")
        .replace("new mexico", "NM")
        .replace("new york", "NY")
        .replace("north carolina", "NC")
        .replace("north dakota", "ND")
        .replace("rhode island", "RI")
        .replace("south carolina", "SC")
        .replace("south dakota", "SD")
        .replace("west virginia", "WV")
        .replace("alabama", "AL")
        .replace("alaska", "AK")
        .replace("arizona", "AZ")
        .replace("arkansas", "AR")
        .replace("california", "CA")
        .replace("colorado", "CO")
        .replace("connecticut", "CT")
        .replace("delaware", "DE")
        .replace("florida", "FL")
        .replace("georgia", "GA")
        .replace("hawaii", "HI")
        .replace("idaho", "ID")
        .replace("illinois", "IL")
        .replace("indiana", "IN")
        .replace("iowa", "IA")
        .replace("kansas", "KS")
        .replace("kentucky", "KY")
        .replace("louisiana", "LA")
        .replace("maine", "ME")
        .replace("maryland", "MD")
        .replace("massachusetts", "MA")
        .replace("michigan", "MI")
        .replace("minnesota", "MN")
        .replace("mississippi", "MS")
        .replace("missouri", "MO")
        .replace("montana", "MT")
        .replace("nebraska", "NE")
        .replace("nevada", "NV")
        .replace("ohio", "OH")
        .replace("oklahoma", "OK")
        .replace("oregon", "OR")
        .replace("pennsylvania", "PA")
        .replace("tennessee", "TN")
        .replace("texas", "TX")
        .replace("utah", "UT")
        .replace("vermont", "VT")
        .replace("virginia", "VA")
        .replace("washington", "WA")
        .replace("wisconsin", "WI")
        .replace("wyoming", "WY")
    )
    return _normalize_lists(text)


def _normalize_unicode_math(text: str) -> str:
    return (
        text.replace("°", "^\\circ")
        .replace("²", "^2")
        .replace("³", "^3")
        .replace("⁴", "^4")
        .replace("⁵", "^5")
        .replace("⁶", "^6")
        .replace("⁷", "^7")
        .replace("⁸", "^8")
        .replace("⁹", "^9")
        .replace("√", "\\sqrt")
        .replace("﹣", "-")
        .replace("﹢", "+")
        .replace("﹦", "=")
        .replace("﹤", "<")
        .replace("﹥", ">")
        .replace("：", ":")
        .replace("π", "\\pi")
    )


_MATH_UNITS_RE = re.compile(r"\^\{?\\circ\}?|\\circ|°|cm[²³]?|mm|km|m[²³]?|kg")


def _strip_numeric(text: str) -> str:
    for ch in "$£€%,":
        text = text.replace(ch, "")
    text = _MATH_UNITS_RE.sub("", _normalize_unicode_math(text))
    return text.replace("\\%", "").replace("\\$", "").strip()


def _soft_numeric_grade(gt_answer: str, pred_answer: str) -> float:
    pred = _strip_numeric(pred_answer)
    gt = _strip_numeric(gt_answer)
    try:
        pv = float(pred)
        gv = float(gt)
    except ValueError:
        return 0.0
    rel_error = abs(pv - gv) / max(abs(gv), 1e-9)
    return max(0.0, 1.0 - rel_error) ** 2


class StringMatchResourcesServer(SimpleResourcesServer):
    config: StringMatchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: StringMatchVerifyRequest) -> StringMatchVerifyResponse:
        text = _extract_last_assistant_text(body)
        extracted = _extract_answer(text, body.extraction_mode)

        reward = 0.0
        if extracted is not None and body.expected_answer:
            reward = _answers_match(extracted, body.expected_answer, body.case_sensitive)

        return StringMatchVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            expected_answer=body.expected_answer,
            extracted_answer=extracted,
        )


if __name__ == "__main__":
    StringMatchResourcesServer.run_webserver()
