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


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace, normalize unicode."""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = " ".join(s.split())
    return s.lower()


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


def _answers_match(extracted: str, expected: str, case_sensitive: bool) -> bool:
    if case_sensitive:
        return extracted.strip() == expected.strip()
    return _normalize(extracted) == _normalize(expected)


class StringMatchResourcesServer(SimpleResourcesServer):
    config: StringMatchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: StringMatchVerifyRequest) -> StringMatchVerifyResponse:
        text = _extract_last_assistant_text(body)
        extracted = _extract_answer(text, body.extraction_mode)

        is_correct = False
        if extracted is not None and body.expected_answer:
            is_correct = _answers_match(extracted, body.expected_answer, body.case_sensitive)

        reward = 1.0 if is_correct else 0.0

        return StringMatchVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            expected_answer=body.expected_answer,
            extracted_answer=extracted,
        )


if __name__ == "__main__":
    StringMatchResourcesServer.run_webserver()
