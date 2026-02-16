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
import contextlib
import json
import logging
import os
from io import StringIO
from typing import Any, ClassVar, Optional

from fastapi import FastAPI
from math_verify import grader
from openai import AsyncOpenAI
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from pydantic import BaseModel, PrivateAttr

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
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import get_response_json

LOG = logging.getLogger(__name__)

# Lock for appending to the judge verification JSONL (when MATH_JUDGE_LOG_DIR is set).
_judge_log_lock: asyncio.Lock | None = None


def _get_judge_log_lock() -> asyncio.Lock:
    global _judge_log_lock
    if _judge_log_lock is None:
        _judge_log_lock = asyncio.Lock()
    return _judge_log_lock


def _get_judge_client_config() -> Optional[tuple[str, str, int, list[str]]]:
    """Read judge server address from env (set by pipeline when using heterogeneous job with judge).

    Returns:
        (model, server_type, port, master_nodes) or None if JUDGE_SERVER_ARGS not set.
    """
    server_args_str = os.environ.get("JUDGE_SERVER_ARGS")
    if not server_args_str:
        LOG.debug("JUDGE_SERVER_ARGS not set; judge will use policy_model (/v1/responses) path.")
        return None
    server_config = json.loads(server_args_str)
    server_type = server_config["server_type"]
    model = server_config["model"]
    n_servers = server_config.get("n_servers", 1)
    port = server_config["port"]
    # When main first: judge is het group 1. When judge first: judge is het group 0. Pipeline injects judge_het_group.
    judge_het_group = server_config.get("judge_het_group", 0)
    master_nodes = []
    for i in range(n_servers):
        het_group = judge_het_group + i
        env_var = f"SLURM_MASTER_NODE_HET_GROUP_{het_group}"
        master_node = os.environ.get(env_var)
        if not master_node:
            LOG.error("Missing env %s for judge server (JUDGE_SERVER_ARGS has judge_het_group=%s, n_servers=%s).", env_var, judge_het_group, n_servers)
            raise RuntimeError(f"Missing {env_var} for judge server (heterogeneous job).")
        master_nodes.append(master_node)
    LOG.info(
        "Judge client config: model=%s, server_type=%s, port=%s, judge_het_group=%s, master_nodes[0]=%s (n_servers=%s).",
        model, server_type, port, judge_het_group, master_nodes[0] if master_nodes else None, n_servers,
    )
    print(
        f"[math_with_judge] Judge config: model={model}, port={port}, master_nodes[0]={master_nodes[0] if master_nodes else None} (n_servers={n_servers})."
    )
    return model, server_type, port, master_nodes


# Hard-coded user prompt for NL math judge (from nemo_skills prompt/config/judge/math.yaml).
# Uses problem, predicted_answer, expected_answer; output format: "Reasoning: ..." and "Judgement: Yes" or "Judgement: No".
JUDGE_USER_PROMPT_NL_MATH: str = """You will be asked to look at the two answers (predicted and expected) to a math problem and to judge whether they are equivalent within the context of the problem.

Please first explain your reasoning in a couple of sentences. Then respond with only Yes or No as your judgement on whether the two answers are the same.
When comparing answers only perform trivial simplifications.

Here are a few examples. Please include both "Reasoning" and "Judgement" in your final response in the same format as below.


Example 1:
Problem: Factor $7x^3 - 21x^2 + 14x$.
Predicted answer: $7x(x - 2)(x - 1)$
Expected answer: $7x(x-1)(x-2)$

Reasoning: The order of the factors does not matter, so the answers are the same.
Judgement: Yes


Example 2:
Problem: A rectangle has a length of 6 meters and a width of 2 meters. If the length is reduced by 3 meters and the width is halved, what is the new area of the rectangle in square meters?
Predicted answer: 3/2
Expected answer: 1.5

Reasoning: 3/2 is the same as 1.5
Judgement: Yes


Example 3:
Problem: Simplify the expression $\\sqrt{{7!}}$, where $n!$ stands for $n\\cdot(n-1)\\cdot(n-2)\\cdots \\cdot 2\\cdot 1$.
Predicted answer: 71
Expected answer: 12\\sqrt{{35}}.

Reasoning: This is non-trivial to simplify, so the answers are different.
Judgement: No


Example 4:
Problem: What is the simplified form of the expression $\\sqrt{{98 x^{{3}} y^{{5}} z}} ?
\\begin{{align*}}
\\text{{A)}} & 2 x y z \\sqrt{{7 x y z}} &
\\text{{B)}} &  7 x^{{2}} y^{{2}} \\sqrt{{2 y z}}
\\\\
\\text{{C)}} & 7 x y^{{2}} \\sqrt{{2 x y z}}  &
\\text{{D)}} &49 x y^{{2}} \\sqrt{{2 x y z}}
\\\\
\\end{{align*}}
Predicted answer: 7 x y^{{2}} \\sqrt{{2 x y z}}
Expected answer: C

Reasoning: Predicted answer is the same as the expected answer choice C.
Judgement: Yes


Example 5:
Problem: A line segment of length $5$ has one endpoint at $(1, 2)$ and the other endpoint at $(4, b)$. Find all possible values of $b$, separated by commas.
Predicted answer:  -2, 6
Expected answer: 6, -2

Reasoning: The order doesn't matter in the context of the problem.
Judgement: Yes


Example 6:
Problem: Solve $\\tan x = \\sin x$ for $0 \\le x \\le 2 \\pi.$  Enter all the solutions, separated by commas.
Predicted answer: 0, \\pi
Expected answer: 0,\\pi,2\\pi.

Reasoning: Number of solutions is different.
Judgement: No


YOUR TASK

Problem: {problem}
Predicted answer: {predicted_answer}
Expected answer: {expected_answer}"""


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
    JUDGE_SYSTEM_MESSAGE: ClassVar[
        str
    ] = """Please act as an impartial judge and evaluate the equivalence of the solutions given by two AI assistants to the mathematical problem displayed below. You will be given AI assistant A's answer and AI assistant B's answer. Your job is to evaluate whether assistant A's answer is equivalent to assistant B's answer.

Consider the mathematical equivalence of the AI assistants' answers above all other considerations. If the problem requests special formatting instructions, you may disregard any formatting considerations when evaluating the answers -- consider only mathematical equivalence.

After evaluating both answers for equivalence, you must output only one of the following choices as your final verdict with a label:

1.  The AI assistants' answers are equivalent: [[A=B]]
2.  The AI assistants' answers are different: [[A!=B]]

Example output: "My final verdict is different [[A!=B]]"."""

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = (
        "<|Problem|>\n{question}\n\n<|Start of Assistant A's Answer|>\n{first_answer}\n<|End of Assistant A's Answer|>\n\n<|Start of Assistant B's Answer|>\n{second_answer}\n<|End of Assistant B's Answer|>"
    )

    JUDGE_EQUAL_LABEL: ClassVar[str] = "[[A=B]]"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = "[[A!=B]]"

    config: LibraryJudgeMathResourcesServerConfig

    # Lazy-initialized when JUDGE_SERVER_ARGS is set (OpenAI-compatible judge).
    _judge_openai_client: Optional[AsyncOpenAI] = PrivateAttr(default=None)
    _judge_model: Optional[str] = PrivateAttr(default=None)
    _judge_first_success_logged: bool = PrivateAttr(default=False)

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
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        combined_response = "".join(assistant_responses)
        (
            reward,
            extracted_answer,
            library_reward,
            judge_evaluations,
        ) = await self._verify_answer(body.question, body.expected_answer, combined_response)
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
        if os.environ.get("JUDGE_SERVER_ARGS"):
            LOG.debug("Using judge path: openai (JUDGE_SERVER_ARGS set, separate judge vLLM).")
            print("[math_with_judge] Using separate judge server (JUDGE_SERVER_ARGS set).")
            return await self._verify_answer_with_judge_openai(
                question, expected_answer, generated_answer
            )
        LOG.debug("Using judge path: policy_model (/v1/responses).")
        print("[math_with_judge] Using policy_model for judge (/v1/responses, JUDGE_SERVER_ARGS not set).")
        # Original path: /v1/responses judge.
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

    async def _verify_answer_with_judge_openai(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[float, list[JudgeEvaluation]]:
        """Use OpenAI-compatible judge (vLLM /v1/chat/completions) when JUDGE_SERVER_ARGS is set."""
        cfg = _get_judge_client_config()
        if not cfg:
            LOG.warning("_verify_answer_with_judge_openai: JUDGE_SERVER_ARGS was set but _get_judge_client_config() returned None; returning reward 0.")
            return 0.0, []
        model, _server_type, port, master_nodes = cfg
        if self._judge_openai_client is None:
            base_url = f"http://{master_nodes[0]}:{port}/v1"
            LOG.info("Initializing OpenAI judge client: base_url=%s, model=%s.", base_url, model)
            print(f"[math_with_judge] Judge client connected: base_url={base_url}, model={model}")
            self._judge_openai_client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
            self._judge_model = model
        user_content = JUDGE_USER_PROMPT_NL_MATH.format(
            problem=question,
            predicted_answer=generated_answer,
            expected_answer=expected_answer,
        )
        try:
            completion = await self._judge_openai_client.chat.completions.create(
                model=self._judge_model,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.0,
            )
        except Exception as e:  # noqa: BLE001
            LOG.exception(
                "Judge request failed (base_url=%s, model=%s): %s",
                self._judge_openai_client.base_url if self._judge_openai_client else "?",
                self._judge_model,
                e,
            )
            raise
        judge_text = (
            completion.choices[0].message.content or ""
        ).strip()
        if not self._judge_first_success_logged:
            print("[math_with_judge] First judge request succeeded (using judge server).")
            self._judge_first_success_logged = True
        # Parse "Judgement: Yes" or "Judgement: No" (case-insensitive); last occurrence wins.
        reward = 0.0
        judgement_parsed = "No"
        if "judgement:" in judge_text.lower():
            last_yes = judge_text.lower().rfind("judgement: yes")
            last_no = judge_text.lower().rfind("judgement: no")
            if last_yes >= 0 and (last_no < 0 or last_yes > last_no):
                reward = 1.0
                judgement_parsed = "Yes"
            else:
                judgement_parsed = "No"
        else:
            LOG.warning(
                "Judge response had no 'Judgement:' line (reward=0). Snippet: %s",
                (judge_text[:500] + "..." if len(judge_text) > 500 else judge_text) or "(empty)",
            )
        # Append to JSONL when MATH_JUDGE_LOG_DIR is set (e.g. by pipeline).
        log_dir = os.environ.get("MATH_JUDGE_LOG_DIR")
        if log_dir:
            record = {
                "problem": question,
                "predicted_answer": generated_answer,
                "expected_answer": expected_answer,
                "judgement_raw": judge_text,
                "judgement": judgement_parsed,
                "reward": int(reward),
            }
            log_path = os.path.join(log_dir, "math_judge_verifications.jsonl")
            try:
                os.makedirs(log_dir, exist_ok=True)
                lock = _get_judge_log_lock()
                async with lock:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as e:
                LOG.warning("Failed to write judge verification to %s: %s", log_path, e)
        # Build one JudgeEvaluation with minimal NeMoGymResponse for consistency.
        out_text = NeMoGymResponseOutputText(annotations=[], text=judge_text)
        out_msg = NeMoGymResponseOutputMessage(
            id="0", content=[out_text], role="assistant", status="completed", type="message"
        )
        judge_response = NeMoGymResponse.model_construct(output=[out_msg])
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content=user_content)],
            model=self._judge_model,
        )
        judge_eval = JudgeEvaluation(responses_create_params=params, response=judge_response)
        return reward, [judge_eval]

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

        response = await self.server_client.post(
            server_name=config.judge_model_server.name,
            url_path="/v1/responses",
            json=responses_create_params,
        )
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))
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

        output_text = last_content.text
        equal_choice_position = output_text.find(self.JUDGE_EQUAL_LABEL)
        not_equal_choice_position = output_text.find(self.JUDGE_NOT_EQUAL_LABEL)

        # The first label that appears in the text is used for the evaluation.
        if equal_choice_position < 0:
            if not_equal_choice_position < 0:
                return False, judge_evaluation
            else:
                return False, judge_evaluation
        else:
            if not_equal_choice_position < 0:
                return True, judge_evaluation
            elif equal_choice_position < not_equal_choice_position:
                return True, judge_evaluation
            else:
                return False, judge_evaluation


if __name__ == "__main__":
    LibraryJudgeMathResourcesServer.run_webserver()
