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
import asyncio
import json
from contextlib import nullcontext
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from openapi_schema_validator import validate as validate_against_schema_openapi
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
from resources_servers.terminus_judge.schemas import TERMINUS_1_SCHEMA, TERMINUS_2_SCHEMA


SCHEMA_MAP = {
    "terminus_1": TERMINUS_1_SCHEMA,
    "terminus_2": TERMINUS_2_SCHEMA,
}


class FailureCode(str, Enum):
    """Enumeration of possible failure reasons."""

    NONE = "none"
    EXPECTED_ANSWER_INVALID = "expected_answer_invalid"  # Ground truth missing or not valid JSON (data issue)
    MODEL_OUTPUT_INVALID = "model_output_invalid"  # Model output empty or not valid JSON
    SCHEMA_CHECK_FAILED = "schema_check_failed"
    TASK_COMPLETE_CHECK_FAILED = "task_complete_check_failed"
    STRING_SIMILARITY_BELOW_THRESHOLD = "string_similarity_below_threshold"  # Similarity score < threshold
    UNKNOWN_HARNESS = "unknown_harness"
    JUDGE_NOT_EQUIVALENT = "judge_not_equivalent"  # Judge returned [[A!=B]]
    JUDGE_PARSING_FAILED = "judge_parsing_failed"  # First judge output invalid (no verdict)
    JUDGE_SWAP_PARSING_FAILED = "judge_swap_parsing_failed"  # Second (swap) judge output invalid
    UNKNOWN_ERROR = "unknown_error"


def extract_keystrokes(data: Dict[str, Any]) -> List[str]:
    """Extract all keystrokes from commands list."""
    commands = data.get("commands", [])
    return [cmd.get("keystrokes", "") for cmd in commands if "keystrokes" in cmd]


def text_similarity(s1: str, s2: str) -> float:
    """Compute similarity ratio between two strings using SequenceMatcher.
    Returns value between 0 (no similarity) and 1 (identical).
    """
    return SequenceMatcher(None, s1, s2).ratio()


def command_similarity(gt: Dict[str, Any], pred: Dict[str, Any], separator: str = "") -> float:
    """
    Compute text similarity between commands in gt and pred.

    Concatenates all commands in sequence before comparing, preserving
    the sequential execution order.

    Args:
        gt: Ground truth dictionary with 'commands' key
        pred: Prediction dictionary with 'commands' key
        separator: String to join commands with (default: empty string)

    Returns:
        Similarity score between 0 (no similarity) and 1 (identical)
    """
    gt_keystrokes = extract_keystrokes(gt)
    pred_keystrokes = extract_keystrokes(pred)

    if not gt_keystrokes and not pred_keystrokes:
        return 1.0  # Both empty = identical

    if not gt_keystrokes or not pred_keystrokes:
        return 0.0  # One empty, one not = no similarity

    # Concatenate commands in sequence
    gt_concat = separator.join(gt_keystrokes)
    pred_concat = separator.join(pred_keystrokes)

    # Compare concatenated command sequences
    return text_similarity(gt_concat, pred_concat)


def check_task_complete(pred: dict, expected_answer: dict) -> bool:
    """Check if task completion flags are properly set."""
    if "task_complete" in expected_answer and expected_answer["task_complete"]:
        if "task_complete" not in pred or not pred["task_complete"]:
            return False
    elif "is_task_complete" in expected_answer and expected_answer["is_task_complete"]:
        if "is_task_complete" not in pred or not pred["is_task_complete"]:
            return False
    return True


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    """Extract the last assistant message text from the response.

    Returns an empty string when no assistant text is available.
    """
    for o in reversed(body.response.output):
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                texts: list[str] = []
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
                return "\n".join(texts).strip()
            elif isinstance(content, str):
                return content.strip()
            break
    return ""


def _extract_expected_answer(req: BaseRunRequest) -> Optional[str]:
    """Extract expected answer from request."""
    if hasattr(req, "expected_answer") and req.expected_answer:
        return str(req.expected_answer)
    md = getattr(req, "metadata", None) or {}
    exp = md.get("expected_answer")
    return str(exp) if exp is not None else None


class TerminusJudgeResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "terminus_judge"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 64
    judge_system_message: Optional[str] = None
    judge_prompt_template_fpath: str = "prompt_templates/terminus_prompt.txt"
    judge_equal_label: str = "[[A=B]]"
    judge_not_equal_label: str = "[[A!=B]]"
    check_twice_swap: bool = False
    reward_if_swap_fails: float = 0.0

    # Verification options (independent, can enable any combination)
    # - Both off: Only schema + task_complete check, reward=1.0 if pass
    # - String sim only: Fast string comparison, reward=1.0 if similarity >= threshold
    # - LLM judge only: Always call LLM judge for semantic equivalence
    # - Both on: String sim as fast-path, LLM judge as fallback when sim fails
    enable_string_similarity: bool = False
    string_similarity_threshold: float = 0.95
    enable_llm_judge: bool = True


class TerminusJudgeRunRequest(BaseRunRequest):
    """Run/verify request payload."""

    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    expected_answer: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    threshold: Optional[float] = None


class TerminusJudgeVerifyRequest(TerminusJudgeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    verdict_label: Optional[str] = None


class TerminusJudgeVerifyResponse(BaseVerifyResponse):
    uuid: Optional[str | int] = None
    expected_answer: str
    model_output: str
    parsed_output: Optional[dict] = None
    similarity_score: Optional[float] = None  # None if string similarity disabled
    schema_check_passed: Optional[bool] = None  # None if not attempted (e.g., JSON parse failed)
    task_complete_check_passed: Optional[bool] = None  # None if schema check failed/skipped
    string_similarity_passed: Optional[bool] = None  # None if string similarity disabled
    judge_passed: Optional[bool] = None  # None if LLM judge disabled
    failure_reason: Optional[FailureCode] = None
    judge_evaluations: list[JudgeEvaluation] = []
    metadata: Optional[dict[str, Any]] = None
    threshold: Optional[float] = None


class TerminusJudgeResourcesServer(SimpleResourcesServer):
    config: TerminusJudgeResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(value=self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = None

        with open(self.config.judge_prompt_template_fpath, "r") as f:
            self._judge_prompt_template = f.read().strip()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: TerminusJudgeVerifyRequest) -> TerminusJudgeVerifyResponse:
        # Initialize response fields
        reward = 0.0
        failure_reason = None
        schema_passed = None
        task_complete_passed = None
        string_similarity_passed = None
        judge_passed = None
        similarity_score = None
        parsed_output = None
        judge_evaluations = []

        # Helper to build response
        def _build_response(expected_str: str, model_output_str: str) -> TerminusJudgeVerifyResponse:
            return TerminusJudgeVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=body.response,
                reward=reward,
                uuid=body.uuid,
                expected_answer=expected_str,
                model_output=model_output_str,
                parsed_output=parsed_output,
                similarity_score=similarity_score,
                schema_check_passed=schema_passed,
                task_complete_check_passed=task_complete_passed,
                string_similarity_passed=string_similarity_passed,
                judge_passed=judge_passed,
                failure_reason=failure_reason,
                judge_evaluations=judge_evaluations,
                metadata=body.metadata,
                threshold=body.threshold,
            )

        # Extract expected answer (ground truth)
        expected = _extract_expected_answer(body)
        if not expected:
            failure_reason = FailureCode.EXPECTED_ANSWER_INVALID
            return _build_response(expected_str="", model_output_str="")

        # Extract model output
        generated = _extract_last_assistant_text(body)
        if not generated:
            failure_reason = FailureCode.MODEL_OUTPUT_INVALID
            return _build_response(expected_str=expected, model_output_str="")

        # Extract thinking tags if present
        text = generated
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        # Parse expected answer (ground truth) - failure here is a data issue
        try:
            expected_dict = json.loads(expected)
        except json.JSONDecodeError:
            failure_reason = FailureCode.EXPECTED_ANSWER_INVALID
            return _build_response(expected_str=expected, model_output_str=text)

        # Parse model prediction
        try:
            pred = json.loads(text)
            parsed_output = pred
        except json.JSONDecodeError:
            failure_reason = FailureCode.MODEL_OUTPUT_INVALID
            return _build_response(expected_str=expected, model_output_str=text)

        # Check harness type
        harness = body.metadata.get("harness", None) if body.metadata else None
        if harness is None or harness not in ["terminus_1", "terminus_2"]:
            failure_reason = FailureCode.UNKNOWN_HARNESS
            return _build_response(expected_str=expected, model_output_str=text)

        # Schema validation (must pass)
        try:
            validate_against_schema_openapi(pred, SCHEMA_MAP[harness])
            schema_passed = True
        except Exception:
            schema_passed = False
            failure_reason = FailureCode.SCHEMA_CHECK_FAILED
            return _build_response(expected_str=expected, model_output_str=text)

        # Task completion check (must pass)
        if check_task_complete(pred, expected_dict):
            task_complete_passed = True
        else:
            task_complete_passed = False
            failure_reason = FailureCode.TASK_COMPLETE_CHECK_FAILED
            return _build_response(expected_str=expected, model_output_str=text)

        # Verification logic (string similarity and LLM judge are independent)
        try:
            need_judge = False

            # Step 1: String similarity check (if enabled)
            if self.config.enable_string_similarity:
                similarity_score = command_similarity(expected_dict, pred)
                threshold = (
                    body.threshold if body.threshold is not None else self.config.string_similarity_threshold
                )

                if similarity_score >= threshold:
                    # String similarity passed - reward 1.0, skip judge
                    string_similarity_passed = True
                    reward = 1.0
                    failure_reason = FailureCode.NONE
                else:
                    # String similarity failed - may need judge
                    string_similarity_passed = False
                    need_judge = self.config.enable_llm_judge
                    if not need_judge:
                        failure_reason = FailureCode.STRING_SIMILARITY_BELOW_THRESHOLD
            else:
                # String similarity disabled (remains None) - need judge if enabled
                need_judge = self.config.enable_llm_judge

            # Step 2: LLM judge evaluation (if needed and enabled)
            if need_judge:
                first_equal, first_eval = await self._generate_judge_evaluation(
                    expected_answer=expected, generated_answer=text
                )
                judge_evaluations.append(first_eval)

                # Check if judge output was valid (has verdict label)
                if first_eval.verdict_label is None:
                    # Judge output invalid - no verdict found (judge_passed stays None)
                    failure_reason = FailureCode.JUDGE_PARSING_FAILED
                    reward = 0.0
                elif first_equal:
                    if self.config.check_twice_swap:
                        second_equal, second_eval = await self._generate_judge_evaluation(
                            expected_answer=text, generated_answer=expected
                        )
                        judge_evaluations.append(second_eval)

                        # Check if second judge output was valid
                        if second_eval.verdict_label is None:
                            # Second (swap) judge output invalid (judge_passed stays None)
                            failure_reason = FailureCode.JUDGE_SWAP_PARSING_FAILED
                            reward = 0.0
                        elif second_equal:
                            judge_passed = True
                            reward = 1.0
                            failure_reason = FailureCode.NONE
                        else:
                            judge_passed = False
                            reward = self.config.reward_if_swap_fails
                            failure_reason = FailureCode.JUDGE_NOT_EQUIVALENT
                    else:
                        judge_passed = True
                        reward = 1.0
                        failure_reason = FailureCode.NONE
                else:
                    # Judge said not equal
                    judge_passed = False
                    failure_reason = FailureCode.JUDGE_NOT_EQUIVALENT
                    reward = 0.0

            # Step 3: If both disabled, schema + task_complete pass = reward 1.0
            if not self.config.enable_string_similarity and not self.config.enable_llm_judge:
                reward = 1.0
                failure_reason = FailureCode.NONE

        except Exception as e:
            failure_reason = FailureCode.UNKNOWN_ERROR
            print(f"DEBUG: Unknown error in verify: {type(e).__name__} {e}", flush=True)

        return _build_response(expected_str=expected, model_output_str=text)

    async def _generate_judge_evaluation(
        self, *, expected_answer: str, generated_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        """Run a single judge evaluation."""
        cfg = self.config
        equal_label = cfg.judge_equal_label
        not_equal_label = cfg.judge_not_equal_label

        responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)

        user_prompt = self._judge_prompt_template.format(
            expected_answer=expected_answer, generated_answer=generated_answer
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
                    "DEBUG: TerminusJudgeResourcesServer: Judge model server timeout",
                    flush=True,
                )
                raise RuntimeError("Judge model server timeout")
            except Exception as e:
                print(
                    f"DEBUG: TerminusJudgeResourcesServer: judge model server HTTP POST error: {type(e).__name__} {e}",
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

        # extract text
        try:
            last_output = judge_response.output[-1]
            if getattr(last_output, "type", None) != "message":
                text = ""
            else:
                last_content = last_output.content[-1]
                text = getattr(last_content, "text", "")
        except Exception:
            text = ""

        # check text for verdict labels using improved parsing
        if text:
            is_equal, verdict_label = self._parse_verdict_from_text(text, equal_label, not_equal_label)

        eval_record.verdict_label = verdict_label
        return is_equal, eval_record

    def _parse_verdict_from_text(
        self, text: str, equal_label: str, not_equal_label: str
    ) -> tuple[bool, str | None]:
        """Parse verdict from judge response text.

        Uses a two-stage strategy:
        1. First look in the last 200 chars (rubrics v4 puts verdict at end after "Final verdict:")
        2. Fall back to full text search if not found

        For the last-part search, we want the LAST occurrence since rubrics format
        puts the final verdict at the very end.

        Returns (is_equal, verdict_label)
        """
        # Strategy 1: Look for verdict in last 200 chars (rubrics puts it at end)
        last_part = text[-200:] if len(text) > 200 else text

        eq_pos = last_part.rfind(equal_label)
        neq_pos = last_part.rfind(not_equal_label)

        if eq_pos >= 0 or neq_pos >= 0:
            # Found at least one verdict in the last part
            # Use the one that appears later (rightmost)
            if eq_pos >= 0 and (neq_pos < 0 or eq_pos > neq_pos):
                return True, equal_label
            elif neq_pos >= 0:
                return False, not_equal_label

        # Strategy 2: Fallback to full text search (first occurrence)
        eq_pos = text.find(equal_label)
        neq_pos = text.find(not_equal_label)

        if eq_pos >= 0 and (neq_pos < 0 or eq_pos < neq_pos):
            return True, equal_label
        elif neq_pos >= 0:
            return False, not_equal_label

        # No verdict found
        return False, None


if __name__ == "__main__":
    TerminusJudgeResourcesServer.run_webserver()
