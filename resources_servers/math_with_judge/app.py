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
import contextlib
import json
import logging
import re
import threading
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar, Optional

# JSONL path for logging every verify: generation (answer part), full_sequence (reasoning + answer), reward, reward_source (llm | math_verify)
ROLLOUT_JSONL_PATH = Path("/scratch/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/wedu/rollout.jsonl")
_ROLLOUT_JSONL_LOCK = threading.Lock()

from fastapi import FastAPI
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from pydantic import BaseModel, ValidationError as PydanticValidationError

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
from nemo_gym.server_utils import get_response_json

logger = logging.getLogger(__name__)
# Truncate long strings for log (e.g. raw vLLM response)
_LOG_MAX_LEN = 800


def _truncate_for_log(s: str, max_len: int = _LOG_MAX_LEN) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... [truncated, total {len(s)} chars]"


class LibraryJudgeMathResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    should_use_judge: bool = True


class LibraryJudgeMathRunRequest(BaseRunRequest):
    question: str
    expected_answer: str


class LibraryJudgeMathVerifyRequest(LibraryJudgeMathRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class LibraryJudgeMathVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]
    library_reward: float
    judge_evaluations: Optional[list[JudgeEvaluation]]


class LibraryJudgeMathResourcesServer(SimpleResourcesServer):
    # These judge messages are adapted from ones used in Arena Hard.
    # https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/utils/judge_utils.py
    # They are intended to serve as example messages for an LLM judge, and have not
    # been customized for a specific judge model.
    # NOTE: We intentionally keep the judge output machine-parseable by requiring
    # a "Judgement: Yes|No" line, and we only do trivial simplifications.
    JUDGE_SYSTEM_MESSAGE: ClassVar[str] = """You will be asked to look at the two answers (Answer A and Answer B) to a math problem and to judge whether they are equivalent within the context of the problem.

Please first explain your reasoning in a couple of sentences. Then respond with only Yes or No as your judgement on whether the two answers are the same.
When comparing answers only perform trivial simplifications (e.g., factor order, 3/2 == 1.5, matching multiple-choice option letters to the shown expression, ordering of a set of solutions). Do NOT perform non-trivial algebraic transformations.

Format your final decision exactly like this (last line):
Judgement: Yes
or
Judgement: No

Here are a few examples.

Example 1:
Problem: Factor $7x^3 - 21x^2 + 14x$.
Answer A: $7x(x - 2)(x - 1)$
Answer B: $7x(x-1)(x-2)$
Reasoning: The order of the factors does not matter, so the answers are the same.
Judgement: Yes

Example 2:
Problem: A rectangle has a length of 6 meters and a width of 2 meters. If the length is reduced by 3 meters and the width is halved, what is the new area of the rectangle in square meters?
Answer A: 3/2
Answer B: 1.5
Reasoning: 3/2 is the same as 1.5.
Judgement: Yes

Example 3:
Problem: Simplify the expression $\\sqrt{7!}$.
Answer A: 71
Answer B: $12\\sqrt{35}$.
Reasoning: This is non-trivial to simplify, so the answers are different.
Judgement: No

Example 4:
Problem: What is the simplified form of the expression $\\sqrt{98 x^{3} y^{5} z}$?
A) $2 x y z \\sqrt{7 x y z}$
B) $7 x^{2} y^{2} \\sqrt{2 y z}$
C) $7 x y^{2} \\sqrt{2 x y z}$
D) $49 x y^{2} \\sqrt{2 x y z}$
Answer A: $7 x y^{2} \\sqrt{2 x y z}$
Answer B: C
Reasoning: Answer A matches option C.
Judgement: Yes

Example 5:
Problem: A line segment of length 5 has one endpoint at (1, 2) and the other endpoint at (4, b). Find all possible values of b, separated by commas.
Answer A: -2, 6
Answer B: 6, -2
Reasoning: The order doesn't matter in the context of the problem.
Judgement: Yes

Example 6:
Problem: Solve $\\tan x = \\sin x$ for $0 \\le x \\le 2 \\pi.$ Enter all the solutions, separated by commas.
Answer A: 0, $\\pi$
Answer B: 0, $\\pi$, $2\\pi$
Reasoning: The number of solutions is different.
Judgement: No
"""

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = (
        "YOUR TASK\n\n"
        "Problem: {question}\n"
        "Answer A: {first_answer}\n"
        "Answer B: {second_answer}\n"
    )

    config: LibraryJudgeMathResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # Use Latex and plain math extraction from predictions
        # https://github.com/huggingface/Math-Verify?tab=readme-ov-file#extraction-targets
        self._library_verifier = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: LibraryJudgeMathVerifyRequest) -> LibraryJudgeMathVerifyResponse:
        q_len = len(body.question or "")
        a_len = len(body.expected_answer or "")
        if not body.question or not body.expected_answer:
            logger.warning(
                "math_with_judge verify: missing question or expected_answer (question_len=%s expected_answer_len=%s). "
                "Upstream data format should provide both.",
                q_len,
                a_len,
            )
        logger.debug(
            "math_with_judge verify: question_len=%s expected_answer_len=%s judge_server=%s",
            q_len,
            a_len,
            self.config.judge_model_server.name,
        )
        assistant_responses = []
        full_sequence_parts = []
        for output_item in body.response.output:
            if getattr(output_item, "type", None) == "reasoning":
                summary = getattr(output_item, "summary", None) or []
                full_sequence_parts.append("".join(getattr(s, "text", "") or "" for s in summary))
                continue
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue
                assistant_responses.append(content_item.text)
                full_sequence_parts.append(content_item.text)

        combined_response = "".join(assistant_responses)
        full_sequence = "".join(full_sequence_parts)
        (
            reward,
            extracted_answer,
            library_reward,
            judge_evaluations,
        ) = await self._verify_answer(body.question, body.expected_answer, combined_response)
        # reward_source: llm = used LLM judge; math_verify = used library only
        reward_source = "llm" if judge_evaluations is not None else "math_verify"
        try:
            with _ROLLOUT_JSONL_LOCK:
                ROLLOUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(ROLLOUT_JSONL_PATH, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "generation": combined_response,
                                "full_sequence": full_sequence,
                                "reward": reward,
                                "reward_source": reward_source,
                                "judge_model_server": (self.config.judge_model_server.name if reward_source == "llm" else None),
                                "library_reward": library_reward,
                                "question": body.question,
                                "expected_answer": body.expected_answer,
                                "extracted_answer": extracted_answer,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        except Exception as e:
            logger.warning("Failed to write rollout jsonl: %s", e)
        logger.debug(
            "math_with_judge verify: reward=%s reward_source=%s judge_server=%s",
            reward,
            reward_source,
            self.config.judge_model_server.name if reward_source == "llm" else None,
        )
        return LibraryJudgeMathVerifyResponse(
            **body.model_dump(),
            reward=reward,
            extracted_answer=extracted_answer,
            library_reward=library_reward,
            judge_evaluations=judge_evaluations,
        )

    async def _verify_answer(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        """Verify the correctness of a generated answer.

        Verify the correctness of the specified model-generated answer to the
        specified question in comparison with the specified expected answer.
        """

        library_reward, extracted_answer = self._verify_answer_with_library(expected_answer, generated_answer)
        if not self.config.should_use_judge or library_reward > 0.5:
            return library_reward, extracted_answer, library_reward, None

        judge_reward, judge_evaluations = await self._verify_answer_with_judge(
            question, expected_answer, generated_answer
        )
        return judge_reward, extracted_answer, library_reward, judge_evaluations

    @classmethod
    @contextlib.contextmanager
    def _mute_output(cls):
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield

    def _verify_answer_with_library(self, expected_answer: str, generated_answer: str) -> tuple[float, Optional[str]]:
        # This functionality is migrated from Nemo RL.
        # https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py
        try:
            ground_truth_parsable = "\\boxed{" + expected_answer + "}"
            with self._mute_output():
                ret_score, extracted_answer = self._library_verifier([ground_truth_parsable], [generated_answer])

            reward = float(ret_score)

            if extracted_answer is not None:
                # Make sure the extracted answer has two elements.
                assert len(extracted_answer) == 2

                extracted_gold, extracted_prediction = extracted_answer

                # Get the extracted answer.
                for pred in extracted_prediction:
                    if any(grader.verify(gold, pred) for gold in extracted_gold):
                        extracted_answer = pred
                        break
                else:
                    # If no match is found, that means all the answers are
                    # incorrect.  The first prediction is used as the extracted
                    # answer.
                    extracted_answer = extracted_prediction[0]

            return reward, extracted_answer

        # It's possible to emit a TimeoutException and that wouldn't be caught since
        # it actually subclasses from BaseException and math-verify itself does not
        # catch it.
        except (Exception, TimeoutException):
            return 0.0, None

    async def _verify_answer_with_judge(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        # The judge is asked to evaluate whether the answers are equal using both
        # orders of the answers, in case there is any positional bias in terms of
        # the order in which the answers are presented to the judge model.
        (
            first_order_equal,
            first_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, expected_answer, generated_answer)
        if not first_order_equal:
            return 0.0, [first_judge_evaluation]

        (
            second_order_equal,
            second_judge_evaluation,
        ) = await self._generate_judge_evaluation(question, generated_answer, expected_answer)
        if second_order_equal:
            reward = 1.0
        else:
            reward = 0.0
        return reward, [first_judge_evaluation, second_judge_evaluation]

    async def _generate_judge_evaluation(
        self, question: str, first_answer: str, second_answer: str
    ) -> tuple[bool, JudgeEvaluation]:
        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)
        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question, first_answer=first_answer, second_answer=second_answer
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(
                role="system",
                content=self.JUDGE_SYSTEM_MESSAGE,
            ),
            NeMoGymEasyInputMessage(
                role="user",
                content=judge_prompt,
            ),
        ]
        # Use same temperature/top_p as policy vLLM worker so the worker's assert passes (shared worker).
        responses_create_params.temperature = 1.0
        responses_create_params.top_p = 1.0

        logger.info(
            "math_with_judge: calling judge server=%s prompt_len=%d",
            config.judge_model_server.name,
            len(judge_prompt),
        )
        response = await self.server_client.post(
            server_name=config.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
        )
        logger.info(
            "math_with_judge: judge HTTP status=%s",
            getattr(response, "status_code", getattr(response, "status", None)),
        )
        raw_json = await get_response_json(response)
        logger.info(
            "math_with_judge: judge raw response (truncated): %s",
            _truncate_for_log(repr(raw_json)),
        )
        try:
            judge_response = NeMoGymResponse.model_validate(raw_json)
        except PydanticValidationError as e:
            logger.error(
                "math_with_judge: judge returned invalid JSON/model. raw_response=%s validation_error=%s",
                _truncate_for_log(repr(raw_json)),
                str(e),
            )
            raise
        judge_evaluation = JudgeEvaluation(responses_create_params=responses_create_params, response=judge_response)

        # Currently, for all the cases in which the response from the LLM judge
        # does not conform to the expected format, the judge's evaluation is
        # treated as if the answers are not equal.  This may not be ideal, but it
        # is intended to minimize the number of failures for verify requests.
        last_output = judge_response.output[-1]
        if last_output.type != "message":
            return False, judge_evaluation

        last_content = last_output.content[-1]
        if last_content.type != "output_text":
            return False, judge_evaluation

        output_text = last_content.text or ""

        # Prefer explicit "Judgement: Yes|No" (last line). Fall back to a trailing Yes/No.
        m = re.search(r"(?is)\bjudg(?:e)?ment\b\s*:\s*(yes|no)\b", output_text)
        if m:
            return (m.group(1).strip().lower() == "yes"), judge_evaluation

        tail = output_text.strip().split()
        if tail:
            last_token = tail[-1].strip().strip('"\'.').lower()
            if last_token in ("yes", "no"):
                return (last_token == "yes"), judge_evaluation

        return False, judge_evaluation


if __name__ == "__main__":
    LibraryJudgeMathResourcesServer.run_webserver()
