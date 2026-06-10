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
import ast
import json
import re
import unicodedata
import warnings
from typing import Any, Literal, Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class PythonListResourcesServerConfig(BaseResourcesServerConfig):
    pass


class PythonListRunRequest(BaseRunRequest):
    expected_answer: str
    extraction_mode: Literal[
        "auto",
        "boxed",
        "final_answer",
        "last_line",
        "full_response",
    ] = "final_answer"
    metadata: Optional[dict[str, Any]] = None


class PythonListVerifyRequest(PythonListRunRequest, BaseVerifyRequest):
    pass


class PythonListVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]
    parsed_expected: Optional[Any]
    parsed_prediction: Optional[Any]


BOXED_START_PATTERN = re.compile(r"\\boxed\s*\{", re.S)
FINAL_ANSWER_PATTERN = re.compile(r"(?i)(?:final\s+answer|answer)\s*[:：]\s*(.+)")
CODE_FENCE_PATTERN = re.compile(r"^```(?:python|json)?\s*(.*?)\s*```$", re.S | re.I)
LATEX_TEXT_WRAP_PATTERN = re.compile(r"\\text\{\s*(.*?)\s*\}", re.S)


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


def _strip_latex_wrappers(text: str) -> str:
    while True:
        m = LATEX_TEXT_WRAP_PATTERN.fullmatch(text)
        if not m:
            return text
        text = m.group(1).strip()


def _strip_code_fence(text: str) -> str:
    m = CODE_FENCE_PATTERN.fullmatch(text.strip())
    if not m:
        return text.strip()
    return m.group(1).strip()


def _strip_answer_punctuation(text: str) -> str:
    text = text.strip()
    while text and text[-1] in ".。":
        text = text[:-1].rstrip()
    return text


def _balanced_group(text: str, start_idx: int, open_ch: str, close_ch: str) -> Optional[str]:
    depth = 0
    in_string: Optional[str] = None
    escape = False
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = None
            continue
        if ch in ("'", '"'):
            in_string = ch
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]
    return None


def _extract_boxed(text: str) -> Optional[str]:
    matches = list(BOXED_START_PATTERN.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    start = match.end() - 1
    group = _balanced_group(text, start, "{", "}")
    if group is None:
        return None
    return _strip_latex_wrappers(group[1:-1].strip())


def _extract_final_answer(text: str) -> Optional[str]:
    matches = FINAL_ANSWER_PATTERN.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip()
    raw = raw.splitlines()[0].strip()
    return _strip_answer_punctuation(raw)


def _extract_last_line(text: str) -> Optional[str]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return _strip_answer_punctuation(line)
    return None


def _extract_sequence_literal(text: str) -> Optional[str]:
    text = _strip_answer_punctuation(_strip_code_fence(text))
    if _parse_sequence(text) is not None:
        return text

    for idx, ch in enumerate(text):
        if ch == "[":
            group = _balanced_group(text, idx, "[", "]")
        elif ch == "(":
            group = _balanced_group(text, idx, "(", ")")
        else:
            continue
        if group is not None and _parse_sequence(group) is not None:
            return group
    return None


def _extract_answer(text: str, mode: str) -> Optional[str]:
    candidates: list[Optional[str]]
    if mode == "boxed":
        candidates = [_extract_boxed(text)]
    elif mode == "final_answer":
        candidates = [_extract_final_answer(text), _extract_boxed(text)]
    elif mode == "last_line":
        candidates = [_extract_last_line(text)]
    elif mode == "full_response":
        candidates = [text.strip()]
    elif mode == "auto":
        candidates = [
            _extract_final_answer(text),
            _extract_boxed(text),
            _extract_last_line(text),
            text.strip(),
        ]
    else:
        candidates = []

    for candidate in candidates:
        if not candidate:
            continue
        literal = _extract_sequence_literal(candidate)
        if literal is not None:
            return literal
        if candidate.strip():
            return _strip_answer_punctuation(candidate)
    return None


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFKC", value).strip()
    return value


def _normalize_sequence(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_sequence(v) for v in value]
    if isinstance(value, list):
        return [_normalize_sequence(v) for v in value]
    return _normalize_scalar(value)


def _parse_sequence(text: str) -> Optional[list[Any]]:
    text = _strip_answer_punctuation(_strip_code_fence(unicodedata.normalize("NFKC", text)))
    if not text:
        return None

    parsers = (ast.literal_eval, json.loads)
    for parser in parsers:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SyntaxWarning)
                value = parser(text)
        except Exception:
            continue
        if isinstance(value, (list, tuple)):
            return _normalize_sequence(value)
    return None


def _word_f1(pred: str, gt: str) -> float:
    pred_words = set(pred.lower().split())
    gt_words = set(gt.lower().split())
    if not pred_words or not gt_words:
        return 0.0
    common = pred_words & gt_words
    if not common:
        return 0.0
    precision = len(common) / len(pred_words)
    recall = len(common) / len(gt_words)
    return 2 * precision * recall / (precision + recall)


def _score_value(pred: Any, gt: Any) -> float:
    if isinstance(pred, str) and isinstance(gt, str):
        return _word_f1(pred, gt)
    if isinstance(pred, list) and isinstance(gt, list):
        return _score_sequence(pred, gt)
    return float(pred == gt)


def _score_sequence(pred: list[Any], gt: list[Any]) -> float:
    if not pred and not gt:
        return 1.0
    correct = sum(_score_value(p, g) for p, g in zip(pred, gt))
    return correct / max(len(pred), len(gt))


def _grade_python_list(expected_answer: str, predicted_answer: str) -> tuple[float, Optional[Any], Optional[Any]]:
    pred_list = _parse_sequence(predicted_answer)
    gt_list = _parse_sequence(expected_answer)
    if pred_list is None or gt_list is None:
        return 0.0, gt_list, pred_list
    return _score_sequence(pred_list, gt_list), gt_list, pred_list


class PythonListResourcesServer(SimpleResourcesServer):
    config: PythonListResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: PythonListVerifyRequest) -> PythonListVerifyResponse:
        text = _extract_last_assistant_text(body)
        extracted = _extract_answer(text, body.extraction_mode)

        reward = 0.0
        parsed_expected = None
        parsed_prediction = None
        if extracted is not None and body.expected_answer:
            reward, parsed_expected, parsed_prediction = _grade_python_list(body.expected_answer, extracted)

        return PythonListVerifyResponse(
            **body.model_dump(
                exclude={
                    "expected_answer",
                    "extracted_answer",
                    "parsed_expected",
                    "parsed_prediction",
                }
            ),
            reward=reward,
            expected_answer=body.expected_answer,
            extracted_answer=extracted,
            parsed_expected=parsed_expected,
            parsed_prediction=parsed_prediction,
        )


if __name__ == "__main__":
    PythonListResourcesServer.run_webserver()
