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
import logging
import re
from io import StringIO
from typing import Any, ClassVar, Dict, List, Optional, Union

from fastapi import FastAPI
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
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
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.server_utils import get_response_json


class LibraryJudgeMathResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    should_use_judge: bool = True
    # Optional path to a YAML file with `system` (optional) and `user`
    # (required) keys providing a custom judge prompt — e.g. an IMO-style
    # autograder. When None, falls back to the Arena-Hard ClassVars below.
    # Matches the pattern used by the omniscience resource server.
    # A missing or empty `system` key omits the system message.
    judge_prompt_path: Optional[str] = None
    # Verdict labels parsed out of the judge's raw output. None => use the
    # Arena-Hard ClassVars.
    judge_equal_label: Optional[str] = None
    judge_not_equal_label: Optional[str] = None
    # When False, skip the second-order (swapped-answers) judge call. Use
    # for unidirectional autograder prompts whose template has a fixed
    # "model solution" / "golden answer" role assignment.
    judge_bidirectional: bool = True
    # Mirror NeMo Skills' judge pipeline exactly for per-rollout parity.
    # When True:
    #   * Strip <think>...</think> (and pre-</think> content) from the
    #     model output before extraction — mirrors Skills' parse_reasoning=True.
    #   * Use the brace-matching _search_boxed extractor instead of
    #     math-verify for the judge input AND for the `extracted_answer`
    #     field (used by majority@k).
    #   * Prefill shortcuts, mirroring nemo_skills.utils.prefill_judgement:
    #     empty extraction -> reward 0 (no LLM call),
    #     extracted.strip() == expected.strip() -> reward 1 (no LLM call).
    #   * Call the LLM judge unconditionally when no prefill fires — the
    #     judge is not gated by symbolic success, matching Skills.
    #   * `reward` == final judge verdict (prefill or LLM), not
    #     `library_reward OR judge`.
    # `library_reward` is still computed for the `symbolic_accuracy`
    # diagnostic metric, so parity runs can compare symbolic drift too.
    skills_parity_mode: bool = False
    # Apply Skills' parse_reasoning=True extraction path to the symbolic-only
    # (no-judge) flow: strip <think>...</think> blocks, then extract via
    # _search_boxed (brace-matching), then run math-verify on the raw boxed
    # LaTeX. Use this for benchmarks whose Skills config is `eval_type=math`
    # with `should_use_judge=false` (e.g. apex-shortlist, aime25, hmmt_feb25).
    # Unlike skills_parity_mode, this does NOT route through the LLM judge —
    # it preserves symbolic-only semantics. If the boxed extraction returns
    # None (e.g. reasoning was truncated), reward is 0 (matches Skills'
    # predicted_answer=None -> no_answer).
    parse_reasoning_like_skills: bool = False


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

        # Symbolic-only parity path for Skills' `eval_type=math`
        # (apex-shortlist / aime25 / hmmt_feb25 etc). Strip <think>...</think>,
        # extract via brace-matcher, run math-verify on the raw boxed LaTeX.
        # Short-circuits to reward=0 when extraction is None, mirroring
        # Skills' predicted_answer=None -> no_answer.
        if self.config.parse_reasoning_like_skills:
            stripped = self._strip_think_tags(generated_answer)
            raw_boxed = self._search_boxed(stripped)
            if raw_boxed is None:
                return 0.0, None, 0.0, None
            library_reward, _ = self._verify_answer_with_library(expected_answer, f"\\boxed{{{raw_boxed}}}")
            return library_reward, raw_boxed, library_reward, None

        library_reward, extracted_answer = self._verify_answer_with_library(expected_answer, generated_answer)

        if self.config.skills_parity_mode:
            return await self._verify_answer_skills_parity(question, expected_answer, generated_answer, library_reward)

        if not self.config.should_use_judge or library_reward > 0.5:
            return library_reward, extracted_answer, library_reward, None

        # Judge input: prefer the raw \boxed{...} LaTeX the model wrote
        # over math-verify's normalized form. math-verify is tuned for
        # numeric/algebraic answers and can silently mangle non-numeric
        # answers (function definitions, sets, conditions) into a
        # degenerate fragment (e.g. "2" extracted from
        #   \boxed{g(x)=2x^3+C \text{ or } g(x)=-2x^3+C, C\in\mathbb{R}}),
        # which the judge then correctly grades as wrong. Falling back
        # to math-verify's extraction preserves the prior behavior when
        # no balanced \boxed{} is present; ultimately falls back to the
        # full generation if neither is available.
        raw_boxed = self._search_boxed(generated_answer)
        judge_answer = raw_boxed if raw_boxed else (extracted_answer if extracted_answer else generated_answer)
        judge_reward, judge_evaluations = await self._verify_answer_with_judge(question, expected_answer, judge_answer)
        return judge_reward, extracted_answer, library_reward, judge_evaluations

    @staticmethod
    def _search_boxed(text: str) -> Optional[str]:
        r"""Brace-matching extractor for \boxed{...} content.

        Returns the raw LaTeX inside the last balanced \boxed{...} in the
        string, preserving the form the model wrote (unlike math-verify,
        which simplifies numerically). Returns None if no balanced
        \boxed{...} (or \fbox{...}) is present.

        Mirrors nemo_skills/evaluation/math_grader.py::search_boxed so that
        Gym can pass the same LaTeX a Skills baseline would pass to its
        judge.
        """
        if "\\boxed" not in text and "\\fbox" not in text:
            return None
        idx = text.rfind("\\boxed")
        if idx < 0:
            idx = text.rfind("\\fbox")
            if idx < 0:
                return None
        num_open = 0
        right_brace_idx = None
        for i in range(idx, len(text)):
            if text[i] == "{":
                num_open += 1
            elif text[i] == "}":
                num_open -= 1
                if num_open == 0:
                    right_brace_idx = i
                    break
        if right_brace_idx is None:
            return None
        retval = text[idx : right_brace_idx + 1]
        for prefix in ("\\boxed{", "\\fbox{"):
            if retval.startswith(prefix) and retval.endswith("}"):
                return retval[len(prefix) : -1]
        return None

    # Think-tag patterns used by _strip_think_tags. Pre-compiled because the
    # stripper runs on every verify() call when skills_parity_mode is on.
    _THINK_PAIR_RE: ClassVar = re.compile(r"<think>.*?</think>", re.DOTALL)
    _THINKING_PAIR_RE: ClassVar = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)
    _PRE_CLOSE_THINK_RE: ClassVar = re.compile(r"^.*?</think>", re.DOTALL)
    _PRE_CLOSE_THINKING_RE: ClassVar = re.compile(r"^.*?</thinking>", re.DOTALL)

    @classmethod
    def _strip_think_tags(cls, text: str) -> str:
        """Mirror Skills' parse_reasoning=True behavior on a single string.

        - Drop paired <think>...</think> / <thinking>...</thinking> blocks.
        - If only a closing </think> (or </thinking>) survives — common when
          the reasoning parser ate the opening tag — drop everything up to
          and including that tag.
        - Leading / trailing whitespace is stripped so downstream extraction
          isn't thrown off by a leading newline.
        """
        text = cls._THINK_PAIR_RE.sub("", text)
        text = cls._THINKING_PAIR_RE.sub("", text)
        text = cls._PRE_CLOSE_THINK_RE.sub("", text, count=1)
        text = cls._PRE_CLOSE_THINKING_RE.sub("", text, count=1)
        return text.strip()

    async def _verify_answer_skills_parity(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        library_reward: float,
    ) -> tuple[float, Optional[str], float, Optional[list[JudgeEvaluation]]]:
        r"""Skills-parity verification path (opt-in via skills_parity_mode).

        Mirrors nemo_skills/inference/llm_math_judge.py end-to-end:
          1. Strip <think>...</think> from the raw generation.
          2. Extract the last balanced \boxed{...} content (raw LaTeX).
          3. Prefill shortcuts (nemo_skills.utils.prefill_judgement):
               empty extraction              -> reward 0, no LLM call
               extracted.strip() == expected -> reward 1, no LLM call
          4. Otherwise call the LLM judge unconditionally — symbolic result
             never short-circuits the judge in this mode.

        The returned `extracted_answer` is the raw-boxed content (matches
        Skills' `predicted_answer` field, used for majority@k grouping).
        `library_reward` continues to reflect math-verify's symbolic check
        so the symbolic_accuracy metric stays a useful diagnostic.

        `judge_evaluations` semantics:
          - LLM was called        -> list populated (length 1 or 2, depending
            on judge_bidirectional).
          - prefill fired          -> empty list [] (non-None). This lets the
            existing _math_score_fn pick up judge_accuracy for prefilled
            rollouts the same way it does for LLM-judged ones (`is not None`
            check). It is the truthful signal: in parity mode, every
            rollout's reward IS a judge verdict, whether from the prefill
            shortcut or an LLM call.
        """
        stripped = self._strip_think_tags(generated_answer)
        extracted = self._search_boxed(stripped)

        # Prefill: empty / missing extraction.
        if not extracted:
            return 0.0, extracted, library_reward, []

        # Prefill: exact string match (Skills strips both sides before comparing).
        if extracted.strip() == expected_answer.strip():
            return 1.0, extracted, library_reward, []

        # LLM judge on every non-prefilled rollout.
        judge_reward, judge_evaluations = await self._verify_answer_with_judge(question, expected_answer, extracted)
        return judge_reward, extracted, library_reward, judge_evaluations

    @classmethod
    @contextlib.contextmanager
    def _mute_output(cls):
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield

    @staticmethod
    def _strip_math_delimiters(s: str) -> str:
        """Strip outer math delimiters from expected answers.

        Many expected_answer values are wrapped in \\(...\\) or $...$,
        which causes the math_verify parser to fail when we wrap them
        in \\boxed{}.  Removing these outer delimiters fixes parsing.
        """
        s = s.strip()
        if s.startswith("\\(") and s.endswith("\\)"):
            s = s[2:-2].strip()
        if s.startswith("$") and s.endswith("$") and len(s) > 1:
            s = s[1:-1].strip()
        return s

    def _verify_answer_with_library(self, expected_answer: str, generated_answer: str) -> tuple[float, Optional[str]]:
        # This functionality is migrated from Nemo RL.
        # https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py
        try:
            stripped = self._strip_math_delimiters(expected_answer)
            ground_truth_parsable = "\\boxed{" + stripped + "}"
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
                    extracted_answer = extracted_prediction[0] if extracted_prediction else None

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

    # ──────────────────────────────────────────────────────────
    # Aggregate metrics overrides
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _math_score_fn(r: dict) -> Dict[str, Union[float, bool]]:
        scores: Dict[str, Union[float, bool]] = {}
        if "library_reward" in r:
            scores["symbolic_accuracy"] = r["library_reward"]
        if "judge_evaluations" in r and r["judge_evaluations"] is not None:
            scores["judge_accuracy"] = r["reward"]
        return scores

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute math-specific metrics: pass@k, majority@k, per-sample statistics."""
        return compute_pass_majority_metrics(
            tasks,
            score_fn=self._math_score_fn,
            answer_key="extracted_answer",
        )[0]

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Select headline metrics for this math benchmark."""
        key: Dict[str, Any] = {}

        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]"))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", exclude_names=["no_answer"]))
        key.update(highest_k_metrics(agent_metrics, "majority@{k}", exclude_names=["no_answer"]))

        return key


if __name__ == "__main__":
    LibraryJudgeMathResourcesServer.run_webserver()
