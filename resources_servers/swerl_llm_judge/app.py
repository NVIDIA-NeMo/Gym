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
from typing import Any, Literal, Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class SWEJudgeResourcesServerConfig(BaseResourcesServerConfig):
    pass


class SWEJudgeRunRequest(BaseRunRequest):
    instance_id: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_split: Optional[str] = None
    expected_answer: Optional[str] = None
    options: Optional[list[dict[str, str]]] = None
    metadata: Optional[dict[str, Any]] = None
    grading_mode: Literal["lenient", "strict"] = "lenient"


class SWEJudgeVerifyRequest(SWEJudgeRunRequest, BaseVerifyRequest):
    pass


class SWEJudgeVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]


LENIENT_SOLUTION_PATTERN = re.compile(r"<solution>\s*(.*?)\s*</solution>", flags=re.DOTALL | re.IGNORECASE)


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    # body.response.output is a list of union types; we only want assistant message texts
    # TODO: @fsoares should we just assume we are always receiving the last message only? Not sure if this is always true.
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            # Each message has content which can be text parts; normalize to string
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


def _extract_options_and_expected(
    body: SWEJudgeRunRequest,
) -> tuple[Optional[list[dict[str, str]]], Optional[str]]:
    return body.options, body.expected_answer


def _get_allowed_letters_from_options(
    options: Optional[list[dict[str, str]]],
) -> set[str]:
    """Collect uppercase option letters from list of single-key dicts."""
    letters: set[str] = set()
    if options:
        for entry in options:
            for k in entry.keys():
                if isinstance(k, str) and len(k) == 1 and k.isalpha():
                    letters.add(k.upper())
                break
    return letters


def _extract_llm_choice_from_solution_block_lenient(llm_output: str, allowed_letters: set[str]) -> Optional[str]:
    # Find the last <solution>...</solution> block to be robust to reasoning
    matches = list(LENIENT_SOLUTION_PATTERN.finditer(llm_output))
    if not matches:
        return None

    # Take the last match (closest to the final answer)
    solution_content = matches[-1].group(1)
    solution_content = solution_content.strip()
    if not solution_content:
        return None

    # Scan for a single letter NOT part of a word
    # Accept e.g. "[A]", "A", but not any letter that's part of a larger word/token.
    for idx, ch in enumerate(solution_content):
        if ch.isalpha():
            left = solution_content[idx - 1] if idx > 0 else None
            right = solution_content[idx + 1] if idx < len(solution_content) - 1 else None
            if (left is None or not left.isalpha()) and (right is None or not right.isalpha()):
                if ch.upper() not in allowed_letters:
                    return None
                return ch.upper()

    return None


def _extract_llm_choice_from_solution_block_strict(llm_output: str, allowed_letters: set[str]) -> Optional[str]:
    """Only accept a (bracketed) single letter in the solution block."""
    patterns = [
        (r"<solution>\s*\[?([A-Za-z])\]?\s*</solution>", re.MULTILINE),
    ]

    for pattern, flags in patterns:
        match = re.search(re.compile(pattern, flags), llm_output)
        if match:
            if match.group(1).upper() not in allowed_letters:
                return None
            return match.group(1).upper()

    return None


class SWEJudgeResourcesServer(SimpleResourcesServer):
    config: SWEJudgeResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: SWEJudgeVerifyRequest) -> SWEJudgeVerifyResponse:
        # Extract the raw assistant text from the NeMo Gym response.
        text = _extract_last_assistant_text(body)
        # Pull options/expected_answer from dataset-style metadata if available
        options, expected_answer = _extract_options_and_expected(body)
        # Derive allowed letters from option keys
        allowed_letters = _get_allowed_letters_from_options(options)
        # Parse the model's choice from the <solution>...</solution> block.
        if body.grading_mode == "lenient":  ## get the first letter from the solution block
            pred_choice = _extract_llm_choice_from_solution_block_lenient(text, allowed_letters)
        elif body.grading_mode == "strict":  ## only one letter is allowed in the solution block
            pred_choice = _extract_llm_choice_from_solution_block_strict(text, allowed_letters)
        else:
            raise ValueError(f"Invalid grading mode: {body.grading_mode}")

        # Normalize the gold choice: required for grading.
        gold = (expected_answer or "").strip().upper()

        is_correct = bool(pred_choice is not None and gold and pred_choice == gold)
        reward = 1.0 if is_correct else 0.0

        return SWEJudgeVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            expected_answer=gold,
            extracted_answer=pred_choice,
        )


if __name__ == "__main__":
    SWEJudgeResourcesServer.run_webserver()


### data needs allowed letters field and grading mode field
