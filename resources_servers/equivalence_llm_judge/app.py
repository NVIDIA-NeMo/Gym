"""
LLM-as-judge resources server.

Compares a model's generated answer to an expected answer using an LLM judge.
The judge prompt is fully configurable via server config.
"""

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
from __future__ import annotations

import re
from typing import Any, Optional

from fastapi import FastAPI
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


class LLMJudgeResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the LLM judge server.

    - judge_model_server: target model server to use as the judge.
    - judge_responses_create_params: base create params; input will be set per request.
    - judge_system_message: optional custom system message for the judge.
    - judge_prompt_template: optional custom prompt template. Supported placeholders:
        {question}, {expected_answer}, {generated_answer}
    - judge_equal_label / judge_not_equal_label: labels the judge must output.
    """

    # Default logical name for this resources server
    name: str = "equivalence_llm_judge"
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

    judge_system_message: Optional[str] = None
    judge_prompt_template: str
    judge_equal_label: str = "[[A=B]]"
    judge_not_equal_label: str = "[[A!=B]]"
    # Optional regex to extract the question from the last user message.
    # If provided and a match is found, the first non-empty capture group is used;
    # otherwise the full match is used.
    question_extract_regex: Optional[str] = None
    # Optional regex to extract the generated response from the last assistant message.
    # The last match is used. If capture groups exist, the first non-empty group is
    # returned; otherwise, the entire last match is used.
    response_extract_regex: Optional[str] = None

    # Swap check: Run second judge pass with swapped expected/generated to detect positional bias
    check_twice_swap: bool = False
    # Reward to assign if the second (swap) pass fails. Defaults to 0.0; can be set to -1.0.
    reward_if_swap_fails: float = 0.0

    # ========================================================================
    # Per-Record Regex Features (OpenQA support)
    # ========================================================================
    # These features enable mixed datasets with different answer formats.
    # They only activate when template_metadata.output_regex is present.
    # Safe to enable by default - falls back to response_extract_regex when
    # no per-record regex is present.

    # [NEW] Enable per-record regex override from template_metadata.output_regex
    use_per_record_regex: bool = True

    # --- The following features ONLY work when use_per_record_regex=True ---

    # [NEW] If set, skip regex extraction when expected_answer length exceeds this threshold.
    # When skipped, the full generation is used instead of extracting with regex.
    # Only applies when per-record regex is present. Set to None to disable.
    extraction_length_threshold: Optional[int] = 120

    # [NEW] If true, when first pass fails, retry with full generation (no regex) for partial credit.
    # Helps recover from regex extraction failures. Only activates when per-record regex exists.
    check_full_generation_on_fail: bool = True

    # [NEW] Reward when full generation check succeeds after first pass fails.
    # Default is 0.5 (partial credit).
    reward_if_full_generation_succeeds: float = 0.5


class LLMJudgeRunRequest(BaseRunRequest):
    """Run/verify request payload.

    Compatible with MCQA-like datasets. Only `expected_answer` is required for
    grading, but `options` and `metadata` are accepted for compatibility.
    """

    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    expected_answer: Optional[str] = None
    options: Optional[list[dict[str, str]]] = None
    metadata: Optional[dict[str, Any]] = None


class LLMJudgeVerifyRequest(LLMJudgeRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    # Extracted verdict token from judge output, e.g., "[[A=B]]" or "[[A!=B]]".
    verdict_label: Optional[str] = None


class LLMJudgeVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    judge_evaluations: list[JudgeEvaluation]


def _extract_last_assistant_text(body: BaseVerifyRequest, extract_regex: Optional[str]) -> str:
    """Extract the last assistant message text from the response.

    - If the assistant message has multiple text blocks, they are joined with newlines.
    - If ``extract_regex`` is provided, the last regex match is used; if capture
      groups exist, the first non-empty group is returned, otherwise the full match.
    - Returns an empty string when no assistant text is available.
    """
    # Return only the last assistant message's text content.
    for o in reversed(body.response.output):
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                # Some providers split a single assistant message into multiple text blocks.
                # Join all text blocks to reconstruct the full message text.
                texts: list[str] = []
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
                text = "\n".join(texts).strip()
                if not text:
                    return text
                if extract_regex:
                    try:
                        matches = list(re.finditer(extract_regex, text, flags=re.MULTILINE | re.DOTALL))
                    except re.error:
                        matches = []
                    if matches:
                        m = matches[-1]
                        groups = m.groups()
                        if groups:
                            for idx in range(1, len(groups) + 1):
                                gv = m.group(idx)
                                if isinstance(gv, str) and gv.strip() != "":
                                    return gv.strip()
                        return m.group(0).strip()
                return text
            elif isinstance(content, str):
                text = content.strip()
                if not text:
                    return text
                if extract_regex:
                    try:
                        matches = list(re.finditer(extract_regex, text, flags=re.MULTILINE | re.DOTALL))
                    except re.error:
                        matches = []
                    if matches:
                        m = matches[-1]
                        groups = m.groups()
                        if groups:
                            for idx in range(1, len(groups) + 1):
                                gv = m.group(idx)
                                if isinstance(gv, str) and gv.strip() != "":
                                    return gv.strip()
                        return m.group(0).strip()
                return text
            break
    return ""


def _extract_expected_answer(req: LLMJudgeRunRequest) -> Optional[str]:
    if req.expected_answer:
        return str(req.expected_answer)
    md = req.metadata or {}
    exp = md.get("expected_answer")
    return str(exp) if exp is not None else None


def _extract_question_text(
    params: NeMoGymResponseCreateParamsNonStreaming,
    question_extract_regex: Optional[str],
) -> str:
    """Extract the question text from the last user message in ``params``.

    - Returns the raw last user message text by default.
    - If ``question_extract_regex`` is provided, the last regex match is used; if
      capture groups exist, the first non-empty group is returned, otherwise the
      full match.
    - Returns an empty string if no user text is available.
    """
    # Return only the last user message's text content.
    last_text: Optional[str] = None
    for m in params.input or []:
        if getattr(m, "role", None) == "user":
            c = getattr(m, "content", None)
            if isinstance(c, str):
                last_text = c
    text = (last_text or "").strip()
    if not text:
        return text
    # Optionally apply a regex to extract a portion of the question text.
    if question_extract_regex:
        try:
            matches = list(re.finditer(question_extract_regex, text, flags=re.MULTILINE | re.DOTALL))
        except re.error:
            matches = []
        if matches:
            m = matches[-1]  # Use the last match
            # Prefer first non-empty capturing group, else the entire match.
            groups = m.groups()
            if groups:
                for idx in range(1, len(groups) + 1):
                    gv = m.group(idx)
                    if isinstance(gv, str) and gv.strip() != "":
                        return gv.strip()
            return m.group(0).strip()
    return text


class LLMJudgeResourcesServer(SimpleResourcesServer):
    """Judge-only verifier using an LLM to compare answers."""

    config: LLMJudgeResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    def _should_skip_for_length(self, body: LLMJudgeVerifyRequest, expected: str) -> bool:
        """Check if length threshold should skip second evaluation (rescue or swap).

        When length exceeds threshold AND per-record regex is present, second eval is redundant:
        - Already using full generation (no regex benefit from rescue)
        - Swap unreliable for long text

        Only applies when there's an actual per-record regex that was skipped due to length.
        """
        if not self.config.use_per_record_regex:
            return False
        if self.config.extraction_length_threshold is None:
            return False
        if len(expected) <= self.config.extraction_length_threshold:
            return False

        # Only skip if there's a per-record regex that would have been skipped
        if hasattr(body, "template_metadata") and isinstance(body.template_metadata, dict):
            if body.template_metadata.get("output_regex"):
                return True  # Per-record regex exists and was skipped due to length

        return False  # No per-record regex, length threshold doesn't apply

    def _get_extraction_regex(self, body: LLMJudgeVerifyRequest, expected: str) -> Optional[str]:
        """Determine which regex to use for extraction, considering per-record overrides and length threshold.

        Returns:
            - str: regex pattern to extract answer (default or per-record override)
            - None: use full generation (when length threshold exceeded)
        """
        extract_regex = self.config.response_extract_regex

        if self.config.use_per_record_regex:
            # Check for per-record regex override
            if hasattr(body, "template_metadata") and isinstance(body.template_metadata, dict):
                regex_override = body.template_metadata.get("output_regex")
                if regex_override:
                    extract_regex = regex_override

                    # Skip per-record regex for long expected answers (return None → full generation)
                    if self.config.extraction_length_threshold is not None:
                        if len(expected) > self.config.extraction_length_threshold:
                            extract_regex = None

        return extract_regex

    def _make_response(
        self, body: LLMJudgeVerifyRequest, expected: str, reward: float, evaluations: list
    ) -> LLMJudgeVerifyResponse:
        """Create verification response with reward and evaluations."""
        payload = body.model_dump()
        payload.pop("expected_answer", None)
        return LLMJudgeVerifyResponse(
            **payload, reward=reward, expected_answer=expected, judge_evaluations=evaluations
        )

    async def _handle_first_pass_failed(
        self,
        body: LLMJudgeVerifyRequest,
        expected: str,
        question: str,
        first_eval,
    ) -> LLMJudgeVerifyResponse:
        """Handle when first judge evaluation fails (returns not equal).

        Options:
        1. Skip rescue for long answers (already using full generation)
        2. Try rescue with full generation (for short answers with regex)
        3. Return immediate failure
        """
        # Skip rescue for long answers - already using full generation
        if self._should_skip_for_length(body, expected):
            return self._make_response(body, expected, reward=0.0, evaluations=[first_eval])

        # Try rescue if configured (only when per-record regex exists and could have failed)
        if (
            self.config.check_full_generation_on_fail
            and self.config.use_per_record_regex
            and hasattr(body, "template_metadata")
            and isinstance(body.template_metadata, dict)
            and body.template_metadata.get("output_regex")
        ):
            # Retry with full generation (no regex) - rescue from regex extraction failure
            generated_full = _extract_last_assistant_text(body, extract_regex=None)
            second_equal, second_eval = await self._generate_judge_evaluation(
                question=question, expected_answer=expected, generated_answer=generated_full
            )

            reward = self.config.reward_if_full_generation_succeeds if second_equal else 0.0
            return self._make_response(body, expected, reward, [first_eval, second_eval])

        # No rescue - immediate failure
        return self._make_response(body, expected, reward=0.0, evaluations=[first_eval])

    async def _handle_first_pass_succeeded(
        self,
        body: LLMJudgeVerifyRequest,
        expected: str,
        question: str,
        generated: str,
        first_eval,
    ) -> LLMJudgeVerifyResponse:
        """Handle when first judge evaluation succeeds (returns equal).

        Options:
        1. Return immediate success (no swap check)
        2. Skip swap for long answers (unreliable for long text)
        3. Run swap check to detect positional bias
        """
        # No swap check configured
        if not self.config.check_twice_swap:
            return self._make_response(body, expected, reward=1.0, evaluations=[first_eval])

        # Skip swap for long answers
        if self._should_skip_for_length(body, expected):
            return self._make_response(body, expected, reward=1.0, evaluations=[first_eval])

        # Run swap check
        second_equal, second_eval = await self._generate_judge_evaluation(
            question=question, expected_answer=generated, generated_answer=expected
        )
        reward = 1.0 if second_equal else self.config.reward_if_swap_fails
        return self._make_response(body, expected, reward, [first_eval, second_eval])

    async def verify(self, body: LLMJudgeVerifyRequest) -> LLMJudgeVerifyResponse:
        """Verify model response by comparing with expected answer using LLM judge.

        Flow:
        1. Extract question and expected answer
        2. Determine extraction regex (per-record override, length threshold)
        3. Extract answer to judge (could be regex-extracted OR full generation)
        4. Run first judge evaluation on extracted answer
        5. Handle failure → rescue with full generation or immediate fail
        6. Handle success → swap check or immediate success
        """
        # Step 1: Extract question and expected answer
        expected = _extract_expected_answer(body) or ""
        question = _extract_question_text(body.responses_create_params, self.config.question_extract_regex)

        # Step 2: Determine extraction regex (None if long answer triggers threshold)
        extract_regex = self._get_extraction_regex(body, expected)

        # Step 3: Extract answer to judge
        # - If extract_regex is not None → regex-extracted answer
        # - If extract_regex is None (long answer) → full generation
        generated = _extract_last_assistant_text(body, extract_regex)

        # Step 4: Run first judge evaluation
        first_equal, first_eval = await self._generate_judge_evaluation(
            question=question, expected_answer=expected, generated_answer=generated
        )

        # Step 5 & 6: Handle result based on first evaluation
        if not first_equal:
            return await self._handle_first_pass_failed(body, expected, question, first_eval)
        else:
            return await self._handle_first_pass_succeeded(body, expected, question, generated, first_eval)

    async def _generate_judge_evaluation(
        self, *, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        cfg = self.config
        equal_label = cfg.judge_equal_label
        not_equal_label = cfg.judge_not_equal_label

        responses_create_params = cfg.judge_responses_create_params.model_copy(deep=True)
        prompt_template = cfg.judge_prompt_template
        system_message = cfg.judge_system_message

        user_prompt = prompt_template.format(
            question=question, expected_answer=expected_answer, generated_answer=generated_answer
        )

        msgs: list[NeMoGymEasyInputMessage] = []
        if system_message is not None and system_message != "":
            msgs.append(NeMoGymEasyInputMessage(role="system", content=system_message))
        msgs.append(NeMoGymEasyInputMessage(role="user", content=user_prompt))
        responses_create_params.input = msgs

        response = await self.server_client.post(
            server_name=cfg.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
        )
        judge_response = NeMoGymResponse.model_validate(await response.json())
        eval_record = JudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response,
            verdict_label=None,
        )

        # Parse the last output; fall back to not-equal if unexpected.
        try:
            last_output = judge_response.output[-1]
            if getattr(last_output, "type", None) != "message":
                return False, eval_record
            last_content = last_output.content[-1]
            text = getattr(last_content, "text", "")
        except Exception:
            return False, eval_record

        eq_pos = text.find(equal_label)
        neq_pos = text.find(not_equal_label)
        if eq_pos < 0 and neq_pos < 0:
            eval_record.verdict_label = None
            return False, eval_record
        if eq_pos >= 0 and (neq_pos < 0 or eq_pos < neq_pos):
            eval_record.verdict_label = equal_label
            return True, eval_record
        eval_record.verdict_label = not_equal_label
        return False, eval_record


if __name__ == "__main__":
    LLMJudgeResourcesServer.run_webserver()
