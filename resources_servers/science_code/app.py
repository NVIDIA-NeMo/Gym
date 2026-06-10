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
"""
ScienceCode LLM-as-judge resources server.

Compares a model's generated scientific Python code to a reference solution using an LLM judge.
Designed for the SciCode benchmark which evaluates scientific computing code generation.
"""

import asyncio
import re
from contextlib import nullcontext
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from resources_servers.science_code.prompts import (
    SCIENCE_CODE_JUDGE_PROMPT_TEMPLATE,
    SCIENCE_CODE_JUDGE_SYSTEM_MESSAGE,
)


class FailureCode(str, Enum):
    """Enumeration of possible failure reasons."""

    NONE = "none"
    NO_RESPONSE = "no_response"
    NO_CODE_EXTRACTED = "no_code_extracted"
    JUDGE_EVALUATION_FAILED = "judge_evaluation_failed"
    UNKNOWN_ERROR = "unknown_error"


def extract_code_from_response(text: str) -> Optional[str]:
    """Extract Python code from model response.

    Attempts to extract code in the following order:
    1. Code wrapped in ```python ... ``` code blocks
    2. Code wrapped in ``` ... ``` code blocks
    3. Raw Python code (functions, classes, imports)

    Returns:
        Extracted Python code or None if no code found.
    """
    if not text:
        return None

    # Try to extract from ```python ... ``` code block
    python_block_pattern = r"```python\s*([\s\S]*?)\s*```"
    matches = re.findall(python_block_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # Try to extract from ``` ... ``` code block
    generic_block_pattern = r"```\s*([\s\S]*?)\s*```"
    matches = re.findall(generic_block_pattern, text)
    if matches:
        # Check if the content looks like Python code
        for match in reversed(matches):
            content = match.strip()
            if re.match(
                r"^\s*(import|from|def|class|#|@|\"|\'|[a-zA-Z_][a-zA-Z0-9_]*\s*=)",
                content,
            ):
                return content

    # Try to find raw Python code (function or class definition)
    # Look for def or class statements
    code_pattern = r"((?:import|from|def|class)\s+[\s\S]*)"
    matches = re.findall(code_pattern, text)
    if matches:
        return matches[-1].strip()

    return None


def _extract_judge_response_text(response: NeMoGymResponse) -> str:
    """Extract the judge's assistant text across all output messages."""
    texts: list[str] = []
    for o in response.output or []:
        if getattr(o, "type", None) != "message":
            continue
        role = getattr(o, "role", None)
        if role is not None and role != "assistant":
            continue
        content = getattr(o, "content", None)
        if isinstance(content, list):
            msg_texts: list[str] = []
            for c in content:
                t = getattr(c, "text", None)
                if isinstance(t, str) and t.strip():
                    msg_texts.append(t.strip())
            if msg_texts:
                texts.append("\n".join(msg_texts))
        elif isinstance(content, str) and content.strip():
            texts.append(content.strip())
    return "\n".join(texts).strip()


class ScienceCodeResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the ScienceCode judge server."""

    name: str = "science_code"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 64
    judge_system_message: str = SCIENCE_CODE_JUDGE_SYSTEM_MESSAGE
    judge_equal_label: str = "[[A=B]]"
    judge_not_equal_label: str = "[[A!=B]]"

    # Swap check: Run second judge pass with swapped expected/generated to detect positional bias
    check_twice_swap: bool = True
    # Reward when the second (swap) pass fails
    reward_if_swap_fails: float = 0.0


class ScienceCodeRunRequest(BaseRunRequest):
    """Run/verify request payload for ScienceCode tasks."""

    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    problem: str  # Scientific computing problem description (required)
    solution: str  # Reference Python solution (required)
    metadata: Optional[dict[str, Any]] = None


class ScienceCodeVerifyRequest(ScienceCodeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    """Record of a single judge evaluation."""

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    verdict_label: Optional[str] = None


class ScienceCodeVerifyResponse(BaseVerifyResponse):
    """Verification response for ScienceCode tasks."""

    uuid: Optional[str | int] = None
    problem: str  # Scientific computing problem
    solution: str  # Reference solution
    model_output: str
    extracted_code: Optional[str] = None
    judge_passed: bool = False
    failure_reason: Optional[FailureCode] = None
    judge_evaluations: list[JudgeEvaluation] = []
    metadata: Optional[dict[str, Any]] = None


class ScienceCodeResourcesServer(SimpleResourcesServer):
    """ScienceCode judge verifier using an LLM to compare scientific code solutions."""

    config: ScienceCodeResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = None

        self._judge_prompt_template = SCIENCE_CODE_JUDGE_PROMPT_TEMPLATE

    async def verify(self, body: ScienceCodeVerifyRequest) -> ScienceCodeVerifyResponse:
        """Verify model response by comparing generated code with reference using LLM judge."""
        # These are required fields, validated by Pydantic
        problem = body.problem
        reference_solution = body.solution

        # Get model output text directly from response
        generated = body.response.output_text or ""

        reward = 0.0
        failure_reason = None
        judge_passed = False
        judge_evaluations = []
        extracted_code = None

        try:
            # Handle empty response gracefully
            if not generated:
                failure_reason = FailureCode.NO_RESPONSE
                reward = 0.0
            else:
                # Extract code from model output
                extracted_code = extract_code_from_response(generated)

                if not extracted_code:
                    failure_reason = FailureCode.NO_CODE_EXTRACTED
                    reward = 0.0
                else:
                    # Run LLM judge evaluation
                    first_equal, first_eval = await self._generate_judge_evaluation(
                        problem=problem,
                        expected_solution=reference_solution,
                        generated_solution=extracted_code,
                    )
                    judge_evaluations.append(first_eval)

                    if first_equal:
                        if self.config.check_twice_swap:
                            # Run swap check
                            second_equal, second_eval = await self._generate_judge_evaluation(
                                problem=problem,
                                expected_solution=extracted_code,
                                generated_solution=reference_solution,
                            )
                            judge_evaluations.append(second_eval)

                            if second_equal:
                                judge_passed = True
                                reward = 1.0
                                failure_reason = FailureCode.NONE
                            else:
                                reward = self.config.reward_if_swap_fails
                                failure_reason = FailureCode.JUDGE_EVALUATION_FAILED
                        else:
                            judge_passed = True
                            reward = 1.0
                            failure_reason = FailureCode.NONE
                    else:
                        failure_reason = FailureCode.JUDGE_EVALUATION_FAILED
                        reward = 0.0

        except Exception as e:
            failure_reason = FailureCode.UNKNOWN_ERROR
            reward = 0.0
            print(f"DEBUG: Unknown error in verify: {type(e).__name__} {e}", flush=True)

        payload = body.model_dump()
        payload.pop("problem", None)
        payload.pop("solution", None)

        return ScienceCodeVerifyResponse(
            **payload,
            reward=reward,
            problem=problem,
            solution=reference_solution,
            model_output=generated,
            extracted_code=extracted_code,
            judge_passed=judge_passed,
            failure_reason=failure_reason,
            judge_evaluations=judge_evaluations,
        )

    async def _generate_judge_evaluation(
        self,
        *,
        problem: str,
        expected_solution: str,
        generated_solution: str,
    ) -> tuple[bool, JudgeEvaluation]:
        """Run a single judge evaluation."""
        cfg = self.config
        equal_label = cfg.judge_equal_label
        not_equal_label = cfg.judge_not_equal_label

        responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)

        user_prompt = self._judge_prompt_template.format(
            problem=problem,
            first_answer=expected_solution,
            second_answer=generated_solution,
        )

        msgs: list[NeMoGymEasyInputMessage] = []
        if cfg.judge_system_message:
            msgs.append(NeMoGymEasyInputMessage(role="system", content=cfg.judge_system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=user_prompt))
        responses_create_params.input = msgs

        ctx = self._judge_endpoint_max_concurrency or nullcontext()
        async with ctx:
            try:
                response = await self.server_client.post(
                    server_name=cfg.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )

                judge_response = NeMoGymResponse.model_validate(await response.json())

            except asyncio.TimeoutError:
                print(
                    "DEBUG: ScienceCodeResourcesServer: Judge model server timeout",
                    flush=True,
                )
                raise RuntimeError("Judge model server timeout")
            except Exception as e:
                print(
                    f"DEBUG: ScienceCodeResourcesServer: judge model server HTTP POST error: {type(e).__name__} {e}",
                    flush=True,
                )
                raise e

        eval_record = JudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
            verdict_label=None,
        )

        verdict_label = None
        is_equal = False

        # Extract text from judge response
        text = _extract_judge_response_text(judge_response)

        # Check text for verdict labels
        if text:
            eq_pos = text.find(equal_label)
            neq_pos = text.find(not_equal_label)

            if eq_pos >= 0 and (neq_pos < 0 or eq_pos < neq_pos):
                verdict_label = equal_label
                is_equal = True
            elif neq_pos >= 0:
                verdict_label = not_equal_label

        eval_record.verdict_label = verdict_label
        return is_equal, eval_record


if __name__ == "__main__":
    ScienceCodeResourcesServer.run_webserver()
