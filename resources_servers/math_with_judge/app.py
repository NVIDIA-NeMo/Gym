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
import math
from collections import Counter
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
from nemo_gym.global_config import TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json


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

        judge_answer = extracted_answer if extracted_answer else generated_answer
        judge_reward, judge_evaluations = await self._verify_answer_with_judge(question, expected_answer, judge_answer)
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

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute math-specific metrics: pass@k, majority@k, per-sample statistics."""
        if not tasks:
            return {}

        k = max(len(rollouts) for rollouts in tasks)
        score_dicts, answers = _extract_scores_and_answers(tasks)
        score_names = sorted({n for ts in score_dicts for s in ts for n in s})

        flat: Dict[str, Any] = {}
        has_answers = any(any(a is not None for a in ta) for ta in answers)

        # Per-sample aggregate (per-rollout-index accuracy across all tasks)
        per_sample = _compute_per_sample(score_dicts, score_names, k)

        # pass@k, pass@1[avg-of-k], majority@k, and std_dev/std_err — all grouped by k_val
        for k_val in range(1, k + 1):
            pass_k, avg_k = _compute_pass_and_avg(score_dicts, score_names, k_val)
            for name, val in pass_k.items():
                flat[f"pass@{k_val}/{name}"] = val
            for name, val in avg_k.items():
                flat[f"pass@1[avg-of-{k_val}]/{name}"] = val

            # std_dev/std_err across runs from per_sample
            for name, values in per_sample.items():
                subset = values[:k_val]
                if len(subset) >= 2:
                    mean = sum(subset) / len(subset)
                    variance = sum((x - mean) ** 2 for x in subset) / (len(subset) - 1)
                    std_dev = math.sqrt(variance)
                    std_err = std_dev / math.sqrt(len(subset))
                    flat[f"pass@1[avg-of-{k_val}]/{name}/std_dev_across_runs"] = std_dev
                    flat[f"pass@1[avg-of-{k_val}]/{name}/std_err_across_runs"] = std_err

            if has_answers:
                maj = _compute_majority_at_k(score_dicts, answers, score_names, k_val)
                for name, val in maj.items():
                    flat[f"majority@{k_val}/{name}"] = val

        flat["per_sample_aggregate"] = per_sample

        # Per-task metrics for group_level_metrics
        flat["per_task_metrics"] = _compute_per_task_metrics(score_dicts, answers, score_names, k)

        return flat

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Select headline metrics for this math benchmark."""
        key: Dict[str, Any] = {}

        # Token usage (not reward — that's redundant with accuracy scores)
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        # Highest-k pass@1[avg-of-*] for all score names including no_answer (no statistics)
        avg_keys = [
            k
            for k in agent_metrics
            if k.startswith("pass@1[avg-of-") and k.count("/") == 1 and "std_dev" not in k and "std_err" not in k
        ]
        highest_k = max(int(k.split("pass@1[avg-of-")[1].split("]")[0]) for k in avg_keys)
        for k in avg_keys:
            if k.startswith(f"pass@1[avg-of-{highest_k}]"):
                key[k] = agent_metrics[k]

        # Highest-k pass@k for accuracy scores only (not no_answer)
        pass_keys = [k for k in agent_metrics if k.startswith("pass@") and "[" not in k and "/no_answer" not in k]
        highest_k = max(int(k.split("@")[1].split("/")[0]) for k in pass_keys)
        for k in pass_keys:
            if k.startswith(f"pass@{highest_k}/"):
                key[k] = agent_metrics[k]

        # Highest-k majority for accuracy scores only (not no_answer)
        maj_keys = [k for k in agent_metrics if k.startswith("majority@") and "/no_answer" not in k]
        highest_k = max(int(k.split("@")[1].split("/")[0]) for k in maj_keys)
        for k in maj_keys:
            if k.startswith(f"majority@{highest_k}/"):
                key[k] = agent_metrics[k]

        return key


# ──────────────────────────────────────────────────────────
# Math metrics computation functions
# ──────────────────────────────────────────────────────────


def _get_score_dict(result: dict) -> Dict[str, Union[float, bool]]:
    """Extract named scores from a math verify response.

    symbolic_accuracy (from library_reward) is the primary score.
    judge_accuracy appears only when the LLM judge actually ran.
    no_answer is 1.0 when no answer could be extracted, 0.0 otherwise.
    reward itself is always available via RewardProfiler's mean/reward.
    """
    scores: Dict[str, Union[float, bool]] = {}
    if "library_reward" in result:
        scores["symbolic_accuracy"] = result["library_reward"]
    if "judge_evaluations" in result and result["judge_evaluations"] is not None:
        scores["judge_accuracy"] = result["reward"]
    if result.get("extracted_answer") is None:
        scores["no_answer"] = 1.0
    else:
        scores["no_answer"] = 0.0
    return {n: int(v) if isinstance(v, bool) else v for n, v in scores.items()}


def _extract_scores_and_answers(
    tasks: List[List[dict]],
) -> tuple[List[List[Dict[str, float]]], List[List[Optional[str]]]]:
    """Extract score dicts and answers for all tasks and rollouts."""
    all_scores: List[List[Dict[str, float]]] = []
    all_answers: List[List[Optional[str]]] = []
    for rollouts in tasks:
        task_scores = [_get_score_dict(r) for r in rollouts]
        task_answers = [r.get("extracted_answer") for r in rollouts]
        all_scores.append(task_scores)
        all_answers.append(task_answers)
    return all_scores, all_answers


def _compute_pass_and_avg(
    all_scores: List[List[Dict[str, float]]], score_names: List[str], k: int
) -> tuple[Dict[str, float], Dict[str, float]]:
    """pass@k (max of first k) and pass@1[avg-of-k] (mean of first k)."""
    pass_k: Dict[str, float] = {}
    avg_k: Dict[str, float] = {}
    for name in score_names:
        pass_vals, avg_vals = [], []
        for task_scores in all_scores:
            vals = [s.get(name) for s in task_scores if name in s]
            if vals:
                first_k = vals[:k]
                avg_vals.append(sum(first_k) / len(first_k))
                if len(vals) >= k:
                    pass_vals.append(max(first_k))
        if pass_vals:
            pass_k[name] = 100.0 * sum(pass_vals) / len(pass_vals)
        if avg_vals:
            avg_k[name] = 100.0 * sum(avg_vals) / len(avg_vals)
    return pass_k, avg_k


def _compute_majority_at_k(
    all_scores: List[List[Dict[str, float]]],
    all_answers: List[List[Optional[str]]],
    score_names: List[str],
    k: int,
) -> Dict[str, float]:
    """majority@k: pick most common answer among first k, use its score."""
    result = {}
    for name in score_names:
        values = []
        for task_scores, task_answers in zip(all_scores, all_answers):
            pairs = [(a, s[name]) for s, a in zip(task_scores[:k], task_answers[:k]) if a is not None and name in s]
            if not pairs:
                continue
            most_common = Counter(a for a, _ in pairs).most_common(1)[0][0]
            values.append(next(score for a, score in pairs if a == most_common))
        if values:
            result[name] = 100.0 * sum(values) / len(values)
    return result


def _compute_per_sample(
    all_scores: List[List[Dict[str, float]]], score_names: List[str], k: int
) -> Dict[str, List[float]]:
    """Element i = pass@1 using only rollout i across all tasks."""
    result: Dict[str, List[float]] = {name: [] for name in score_names}
    for sample_idx in range(k):
        for name in score_names:
            vals = [ts[sample_idx][name] for ts in all_scores if sample_idx < len(ts) and name in ts[sample_idx]]
            if vals:
                result[name].append(100.0 * sum(vals) / len(vals))
    return {name: values for name, values in result.items() if values}


def _compute_per_task_metrics(
    all_scores: List[List[Dict[str, float]]],
    all_answers: List[List[Optional[str]]],
    score_names: List[str],
    k: int,
) -> List[Dict[str, Any]]:
    """Per-task evaluation metrics: pass@k, majority@k, no_answer for each task."""
    per_task = []
    for task_idx, (task_scores, task_answers) in enumerate(zip(all_scores, all_answers)):
        entry: Dict[str, Any] = {TASK_INDEX_KEY_NAME: task_idx}
        n = len(task_scores)

        # pass@k per task: max of first k_val scores
        # pass@1[avg-of-k] per task: mean of first k_val scores
        for name in score_names:
            vals = [s.get(name) for s in task_scores if name in s]
            if not vals:
                continue
            for k_val in range(1, min(k, n) + 1):
                first_k = vals[:k_val]
                entry[f"pass@{k_val}/{name}"] = max(first_k)
                entry[f"pass@1[avg-of-{k_val}]/{name}"] = sum(first_k) / len(first_k)

        # majority@k per task
        for k_val in range(1, min(k, n) + 1):
            for name in score_names:
                pairs = [
                    (a, s[name])
                    for s, a in zip(task_scores[:k_val], task_answers[:k_val])
                    if a is not None and name in s
                ]
                if pairs:
                    most_common = Counter(a for a, _ in pairs).most_common(1)[0][0]
                    entry[f"majority@{k_val}/{name}"] = next(sc for a, sc in pairs if a == most_common)

        per_task.append(entry)
    return per_task


if __name__ == "__main__":
    LibraryJudgeMathResourcesServer.run_webserver()
