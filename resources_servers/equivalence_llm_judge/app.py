"""
LLM-as-judge resources server.

Compares a model's generated answer to an expected answer using an LLM judge.
The judge prompt is fully configurable via server config.
"""

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from pydantic import BaseModel

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
    # If true, perform a second judge pass swapping expected and generated answers
    # to reduce potential positional bias. Default is false for speed.
    check_twice_swap: bool = False
    # Reward to assign if the second (swap) pass fails. Defaults to 0.0; can be set to -1.0.
    reward_if_swap_fails: float = 0.0


class LLMJudgeRunRequest(BaseRunRequest):
    """Run/verify request payload.

    Compatible with MCQA-like datasets. Only `expected_answer` is required for
    grading, but `options` and `metadata` are accepted for compatibility.
    """

    uuid: Optional[str] = None
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

    async def verify(self, body: LLMJudgeVerifyRequest) -> LLMJudgeVerifyResponse:
        expected = _extract_expected_answer(body) or ""
        question = _extract_question_text(body.responses_create_params, self.config.question_extract_regex)
        generated = _extract_last_assistant_text(body, self.config.response_extract_regex)

        # Run judge twice to mitigate positional or presentation bias by swapping orders.
        first_equal, first_eval = await self._generate_judge_evaluation(
            question=question, expected_answer=expected, generated_answer=generated
        )
        if not first_equal:
            reward = 0.0
            payload = body.model_dump()
            # Avoid duplicate field when constructing response
            payload.pop("expected_answer", None)
            return LLMJudgeVerifyResponse(
                **payload, reward=reward, expected_answer=expected, judge_evaluations=[first_eval]
            )

        # If first pass says equal, optionally confirm with a second pass (swap answers).
        if not self.config.check_twice_swap:
            payload = body.model_dump()
            payload.pop("expected_answer", None)
            return LLMJudgeVerifyResponse(
                **payload, reward=1.0, expected_answer=expected, judge_evaluations=[first_eval]
            )

        second_equal, second_eval = await self._generate_judge_evaluation(
            question=question, expected_answer=generated, generated_answer=expected
        )
        # If they are both equal, we give a reward of 1.0; otherwise use configured fallback.
        # User has to expect this on the training side to discard the data points if negative.
        reward = 1.0 if second_equal else self.config.reward_if_swap_fails
        payload = body.model_dump()
        payload.pop("expected_answer", None)
        return LLMJudgeVerifyResponse(
            **payload, reward=reward, expected_answer=expected, judge_evaluations=[first_eval, second_eval]
        )

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
        judge_response = NeMoGymResponse.model_validate(response.json())
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
